//===-- LoopUnrollASM.cpp - Loop Unrolling at Assembly Level -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass traverses machine loops and optimizes assembly for tight loops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-asm"

// totals
STATISTIC(NumLoopsDetected, "Number of tight loops detected by LoopUnrollASM");
STATISTIC(NumLoopsUnrolled, "Number of tight loops unrolled by LoopUnrollASM");
STATISTIC(NumLoopsUnrolledCondUncond, "Number of tight loops unrolled with cond+uncond terminators");
STATISTIC(NumLoopsUnrolledMultiCondExit, "Number of tight loops unrolled with multi-cond-exit+backedge pattern");
STATISTIC(NumLoopsNotEnoughBubbles, "Number of tight loops skipped (not enough bubbles)");
STATISTIC(NumAddedInsts, "Number of added instruction by loop-unroll-asm pass");
STATISTIC(NumInnermostLoops, "Number of inner-most loops");
// skipped; inner denotes innermost.
STATISTIC(NumInnerLoopsNotSingleBB, "Number of inner loops skipped (Header != Latch)");
STATISTIC(NumInnerLoopsMultipleTerminators2, "Number of inner loops skipped (2 terminators)");
STATISTIC(NumInnerLoopsMultipleTerminators3, "Number of inner loops skipped (3 terminators)");
STATISTIC(NumInnerLoopsMultipleTerminators4Plus, "Number of inner loops skipped (4+ terminators)");
STATISTIC(NumInnerLoopsInvalid, "Number of inner loops skipped (invalid)");
STATISTIC(NumInnerLoopsBranchUnconditional, "Number of inner loops skipped (unconditional branch)");
STATISTIC(NumInnerLoopsBranchIndirect, "Number of inner loops skipped (indirect branch)");
STATISTIC(NumInnerLoopsBranchConditionalNoBackedge, "Number of inner loops skipped (conditional branch without backedge)");
STATISTIC(NumInnerLoopsHasAtomicOps, "Number of inner loops skipped (has atomic ops)");
STATISTIC(NumInnerLoopsHasFcsel, "Number of inner loops skipped (has FCSEL inst)");
STATISTIC(NumInnerLoopsHasInternalBranch, "Number of inner loops skipped (has internal branch)");
STATISTIC(NumInnerLoopsTooManyBlocks, "Number of inner loops skipped (too many blocks)");
STATISTIC(NumInnerLoopsTooManyInsts, "Number of inner loops skipped (too many instructions)");
STATISTIC(NumInnerLoopsBranchPrepFailure, "Number of loops skipped (branch analysis or condition inversion failed)");
// dev stats
STATISTIC(NumInnerLoops_BackedgeFallthruHeader, "Number of inner loops where Backedge branch fallsthrough into Loop Header");
STATISTIC(NumInnerLoops_InvalidUncondExit, "Number of inner loops ending with weird unconditional exit branch");
STATISTIC(NumInnerLoops_LastInstNotBranch, "Number of inner loops where last instruction is not a branch");

static cl::opt<unsigned> LoopUnrollASMMaxInsts(
    "loop-unroll-asm-max-insts",
    cl::desc("Maximum number of instructions in a loop for LoopUnrollASM to process"),
    cl::init(46), cl::Hidden);

static cl::opt<float> LoopUnrollASMFetchBubblesThreshold(
    "loop-unroll-asm-fetch-bubbles-threshold",
    cl::desc("Threshold for fetch bubbles ratio to trigger loop unrolling"),
    cl::init(0.20f), cl::Hidden);

static cl::opt<unsigned> LoopUnrollASMEnable(
    "loop-unroll-asm-enable",
    cl::desc("Bitmask to enable each pattern in LoopUnrollASM "
             "(bit 0: One_Backedge, bit 1: Two_Backedge_Uncond, bit 2: Multi_CondExit_Backedge)"),
    cl::init(0x3), cl::Hidden);

namespace {
enum TerminatorPattern {
  Nonsupported = 0,
  One_Backedge,
  Two_Backedge_Uncond,
  Multi_CondExit_Backedge
};

struct BackedgeInfo {
  MachineInstr *Branch;
  TerminatorPattern Pattern;
  MachineBasicBlock *ExitBlock;
  SmallVector<MachineOperand, 4> Cond;
  SmallVector<MachineOperand, 4> InvertedCond;
  bool EndsWithUncondExit;

  BackedgeInfo(MachineInstr *Branch, TerminatorPattern Pattern, MachineBasicBlock *ExitBlock,
               const SmallVectorImpl<MachineOperand> &Cond,
               const SmallVectorImpl<MachineOperand> &InvertedCond,
               bool EndsWithUncondExit = false)
    : Branch(Branch), Pattern(Pattern), ExitBlock(ExitBlock), Cond(Cond.begin(), Cond.end()),
      InvertedCond(InvertedCond.begin(), InvertedCond.end()), EndsWithUncondExit(EndsWithUncondExit) {}
};

class LoopUnrollASM : public MachineFunctionPass {
public:
  static char ID;
  LoopUnrollASM() : MachineFunctionPass(ID) {
    initializeLoopUnrollASMPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Loop Unroll ASM"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    // Don't preserve MachineLoopInfo since we're changing loop structure
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const TargetInstrInfo *TII = nullptr;

  // Static helper function to calculate bubbles
  static unsigned calculateBubbles(unsigned LoopCount) {
    const unsigned MachineWidth = 10;
    unsigned Remainder = LoopCount % MachineWidth;
    return Remainder ? MachineWidth - Remainder : 0;
  }

  static bool isCompareBranchFusion(const MachineInstr *CompareInst,
                                    const MachineInstr *BranchInst,
                                    const TargetInstrInfo *TII);
  bool isPostIndexMemOp(const MachineInstr &MI);
  bool isLoadStorePair(const MachineInstr &MI);
  static bool isLoopSimplifyForm(const MachineLoop *Loop);
  static void debugPrintLoopInfo(MachineFunction &MF, const MachineLoop *Loop,
                          StringRef Prefix, MachineBasicBlock *ExitBlock = nullptr);
  static MachineInstr* findLoopBackedgeBranch(const MachineLoop* Loop);
  bool processLoop(MachineLoop *Loop, MachineFunction &MF);
  bool processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                        MachineBasicBlock *Header, unsigned LoopCount,
                        BackedgeInfo &Backedge);
  unsigned findBestUnrollCount(unsigned LoopCount, unsigned Bubbles,
                               unsigned MachineWidth);
  void duplicateLoopBody(MachineLoop *Loop, unsigned UnrollFactor,
                         const BackedgeInfo &Backedge);
};
} // end anonymous namespace

char LoopUnrollASM::ID = 0;
char &llvm::LoopUnrollASMID = LoopUnrollASM::ID;

INITIALIZE_PASS_BEGIN(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)

bool LoopUnrollASM::runOnMachineFunction(MachineFunction &MF) {
  if (!LoopUnrollASMEnable)
    return false;

  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  TII = MF.getSubtarget().getInstrInfo();

  if (MLI.empty())
    return false;

  bool Changed = false;

  // Process all loops, starting with innermost ones
  for (MachineLoop *Loop : MLI) {
    Changed |= processLoop(Loop, MF);
  }

  return Changed;
}

/// Find the backedge branch instruction for a given machine loop
/// Returns nullptr if no backedge branch is found or if the loop structure is invalid
MachineInstr* LoopUnrollASM::findLoopBackedgeBranch(const MachineLoop* Loop) {
  MachineBasicBlock* Header = Loop->getHeader();

  // Find all backedge blocks (blocks in the loop that branch back to header)
  SmallVector<MachineBasicBlock*, 4> BackedgeBlocks;
  for (MachineBasicBlock* Block : Loop->blocks())
    // Check if this block has a successor that is the loop header
    for (MachineBasicBlock* Succ : Block->successors())
      if (Succ == Header) {
        BackedgeBlocks.push_back(Block);
        break; // A block can only have one edge to the header
      }
  if (BackedgeBlocks.empty()) {
    LLVM_DEBUG(dbgs() << "Warning: No backedge blocks found for loop\n");
    return nullptr;
  }

  // Handle multiple backedge blocks (choose the one with highest execution frequency
  // or use heuristics to pick the "main" backedge)
  MachineBasicBlock* MainBackedgeBlock = nullptr;
  if (BackedgeBlocks.size() == 1) {
    MainBackedgeBlock = BackedgeBlocks[0];
  } else {
    // Multiple backedges - use heuristics to find the main one
    // Prefer the block that:
    // 1. Is a latch block (if available)
    // 2. Has the highest block frequency (if profile info available)
    // 3. Is the lexicographically last block (fallback)

    MachineBasicBlock* Latch = Loop->getLoopLatch();
    if (Latch && llvm::find(BackedgeBlocks, Latch) != BackedgeBlocks.end()) {
      MainBackedgeBlock = Latch;
    } else {
      // Fallback: choose the last backedge block
      MainBackedgeBlock = BackedgeBlocks.back();
      LLVM_DEBUG(dbgs() << "Warning: Multiple backedge blocks, choosing block "
                        << MainBackedgeBlock->getNumber() << "\n");
    }
  }
  if (!MainBackedgeBlock)
    return nullptr;

  // Find the terminating branch instruction in the backedge block
  MachineInstr* BackedgeBranch = nullptr;
  // Look for the last branching instruction in the block
  for (auto I = MainBackedgeBlock->rbegin(); I != MainBackedgeBlock->rend(); ++I) {
    MachineInstr& MI = *I;

    if (!MI.isBranch())
      continue;

    // Verify this branch actually targets the loop header
    bool BranchesToHeader = false;
    if (MI.isConditionalBranch() || MI.isUnconditionalBranch()) {
      for (const MachineOperand& MO : MI.operands()) {
        if (MO.isMBB() && MO.getMBB() == Header) {
          BranchesToHeader = true;
          break;
        }
      }
      // Also check fall-through case for conditional branches
      if (!BranchesToHeader && MI.isConditionalBranch() &&
          (MainBackedgeBlock->getNextNode() == Header)) {
        BranchesToHeader = true;
        LLVM_DEBUG(dbgs() << "  backedge falls-through loop header\n");
        ++NumInnerLoops_BackedgeFallthruHeader;
      }
    }

    if (BranchesToHeader) {
      BackedgeBranch = &MI;
      break;
    }
  }

  LLVM_DEBUG(if (!BackedgeBranch) dbgs() << "Warning: No branch found that targets loop header\n");

  return BackedgeBranch;
}

/// Check if two consecutive instructions form a Compare-Branch Fusion sequence
/// according to ARM CPU architecture specifications.
/// Returns true if fusion can occur, false otherwise.
bool LoopUnrollASM::isCompareBranchFusion(const MachineInstr *CompareInst,
                                          const MachineInstr *BranchInst,
                                          const TargetInstrInfo *TII) {
  if (!CompareInst || !BranchInst || !TII)
    return false;

  // Branch must immediately follow the compare instruction
  if (CompareInst->getParent() != BranchInst->getParent())
    return false;

  StringRef CompareOpName = TII->getName(CompareInst->getOpcode());
  StringRef BranchOpName = TII->getName(BranchInst->getOpcode());

  // Check for shifted inputs in the comparison - fusion cannot occur with shifted inputs
  // Look for shift operands in the compare instruction
  for (const MachineOperand &MO : CompareInst->operands()) {
    if (MO.isImm()) {
      // Check if this immediate could be a shift amount
      // On AArch64, shift operands are typically encoded in specific patterns
      // This is a heuristic - a more precise check would require opcode-specific knowledge
      continue;
    }
  }

  // Pattern detection based on opcode names (AArch64-specific)
  // Case 1: Flag-producing comparison followed by conditional branch
  // Comparisons: CMN, CMP, TEQ, TST, ADDS, SUBS, ANDS, BICS (excluding FCMP)
  bool isFlagProducingCompare =
      (CompareOpName.contains("CMP") && !CompareOpName.contains("FCMP")) ||   // CMP variants (excluding FCMP)
      CompareOpName.contains("CMN") ||   // CMN variants
      CompareOpName.contains("TST") ||   // TST variants
      CompareOpName.contains("TEQ") ||   // TEQ variants
      CompareOpName == "ADDS" || CompareOpName.starts_with("ADDSWr") || CompareOpName.starts_with("ADDSXr") ||
      CompareOpName == "SUBS" || CompareOpName.starts_with("SUBSWr") || CompareOpName.starts_with("SUBSXr") ||
      CompareOpName == "ANDS" || CompareOpName.starts_with("ANDSWr") || CompareOpName.starts_with("ANDSXr") ||
      CompareOpName == "BICS" || CompareOpName.starts_with("BICSWr") || CompareOpName.starts_with("BICSXr");

  // Conditional branch (B.cond)
  bool isConditionalBranch = BranchInst->isConditionalBranch() &&
                             (BranchOpName.starts_with("B") || BranchOpName.contains("Bcc"));

  if (isFlagProducingCompare && isConditionalBranch) {
    LLVM_DEBUG(dbgs() << "    Observed Compare-Branch Fusion (Case 1): "
                      << CompareOpName << " + " << BranchOpName << "\n");
    return true;
  }

  // Case 2: ALU operation followed by CBZ/CBNZ based on result register
  // ALU operations: ADD, SUB, AND, BIC, ORN, ORR, EOR
  bool isALUOp =
      (CompareOpName.contains("ADD") && !CompareOpName.contains("ADDS")) ||  // ADD but not ADDS
      (CompareOpName.contains("SUB") && !CompareOpName.contains("SUBS")) ||  // SUB but not SUBS
      (CompareOpName.contains("AND") && !CompareOpName.contains("ANDS")) ||  // AND but not ANDS
      (CompareOpName.contains("BIC") && !CompareOpName.contains("BICS")) ||  // BIC but not BICS
      CompareOpName.contains("ORN") ||
      CompareOpName.contains("ORR") ||
      CompareOpName.contains("EOR");

  // CBZ or CBNZ
  bool isCBZorCBNZ = BranchOpName.starts_with("CBZ") || BranchOpName.starts_with("CBNZ");

  if (isALUOp && isCBZorCBNZ) {
    // Verify that CBZ/CBNZ operates on the result register produced by the ALU op
    // Get the result register from ALU operation (typically first def operand)
    Register ALUResultReg;
    for (const MachineOperand &MO : CompareInst->operands()) {
      if (MO.isReg() && MO.isDef()) {
        ALUResultReg = MO.getReg();
        break;
      }
    }

    // Get the register tested by CBZ/CBNZ (typically first use operand)
    Register BranchTestReg;
    for (const MachineOperand &MO : BranchInst->operands()) {
      if (MO.isReg() && MO.isUse()) {
        BranchTestReg = MO.getReg();
        break;
      }
    }

    if (ALUResultReg.isValid() && BranchTestReg.isValid() &&
        ALUResultReg == BranchTestReg) {
      LLVM_DEBUG(dbgs() << "    Observed Compare-Branch Fusion (Case 2): "
                        << CompareOpName << " + " << BranchOpName << "\n");
      return true;
    }
  }

  return false;
}

/// Check if a memory operation is a post-index operation.
/// Post-index operations update the base register and are counted with weight 2.
/// Returns true if the instruction is a post-index memory operation.
bool LoopUnrollASM::isPostIndexMemOp(const MachineInstr &MI) {
  if (!MI.mayLoadOrStore() || MI.isTerminator())
    return false;

  // Check if the instruction modifies its base register
  // This is a heuristic for post-index addressing modes
  bool isPostIndex = false;

  // Get the opcode name to check for explicit post-index patterns
  StringRef OpcodeName = TII->getName(MI.getOpcode());

  // Common patterns for post-index instructions on AArch64
  if (OpcodeName.contains("_POST") ||      // Explicit POST suffix
      OpcodeName.contains("PostIndex") ||   // PostIndex suffix
      OpcodeName.contains_insensitive("writeback")) { // Writeback variant
    isPostIndex = true;
  } else {
    // Check if the instruction has both load/store and register def
    // This catches instructions that modify their base register
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && !MO.isImplicit()) {
        // Check if this defined register is also used as a base
        for (const MachineOperand &UseMO : MI.operands()) {
          if (UseMO.isReg() && UseMO.isUse() &&
              UseMO.getReg() == MO.getReg()) {
            // Same register is both used and defined - likely post-index
            isPostIndex = true;
            break;
          }
        }
        if (isPostIndex)
          break;
      }
    }
  }

  if (isPostIndex) {
    LLVM_DEBUG(dbgs() << "    Observed post-index instruction in: " << OpcodeName << "\n");
  }

  return isPostIndex;
}

/// Check if a memory operation is a load-pair or store-pair instruction.
/// Load/store pair instructions operate on two registers and are counted with weight 2.
/// Returns true if the instruction is a load-pair or store-pair operation.
bool LoopUnrollASM::isLoadStorePair(const MachineInstr &MI) {
  if (!MI.mayLoadOrStore() || MI.isTerminator())
    return false;

  // Get the opcode name to check for pair patterns
  StringRef OpcodeName = TII->getName(MI.getOpcode());

  // Common patterns for load/store pair instructions on AArch64
  // LDP - Load Pair of Registers
  // STP - Store Pair of Registers
  // LDNP - Load Pair Non-temporal
  // STNP - Store Pair Non-temporal
  bool isPairOp = OpcodeName.starts_with("LDP") ||   // LDP variants (LDPWi, LDPXi, LDPSi, LDPDi, LDPQi, etc.)
                  OpcodeName.starts_with("STP") ||   // STP variants (STPWi, STPXi, STPSi, STPDi, STPQi, etc.)
                  OpcodeName.starts_with("LDNP") ||  // LDNP variants (LDNPWi, LDNPXi, LDNPSi, LDNPDi, LDNPQi, etc.)
                  OpcodeName.starts_with("STNP");    // STNP variants (STNPWi, STNPXi, STNPSi, STNPDi, STNPQi, etc.)

  if (isPairOp) {
    LLVM_DEBUG(dbgs() << "    Observed load/store pair instruction: " << OpcodeName << "\n");
  }

  return isPairOp;
}

/// Check if a MachineLoop is in loop simplify form.
/// A loop in simplify form has:
/// - A preheader (single predecessor outside the loop)
/// - A single latch (single block with backedge to header)
/// - Dedicated exits (all exit blocks have all predecessors inside the loop)
bool LoopUnrollASM::isLoopSimplifyForm(const MachineLoop *Loop) {
  // Check for preheader and single latch
  if (!Loop->getLoopPreheader() || !Loop->getLoopLatch())
    return false;

  // Check for dedicated exits: all exit blocks must have all predecessors inside the loop
  SmallVector<MachineBasicBlock*, 8> ExitBlocks;
  Loop->getExitBlocks(ExitBlocks);

  for (MachineBasicBlock *ExitBB : ExitBlocks) {
    for (MachineBasicBlock *Pred : ExitBB->predecessors()) {
      if (!Loop->contains(Pred)) {
        // Found a predecessor outside the loop - not dedicated exits
        return false;
      }
    }
  }

  return true;
}

bool LoopUnrollASM::processLoop(MachineLoop *Loop, MachineFunction &MF) {
  // Process inner loops first (depth-first)
  bool Changed = false;
  for (MachineLoop *SubLoop : *Loop) {
    Changed |= processLoop(SubLoop, MF);
  }

  MachineBasicBlock *Header = Loop->getHeader();
  LLVM_DEBUG(debugPrintLoopInfo(MF, Loop, "Examining"));
  // Only process innermost loops
  if (!Loop->getSubLoops().empty()) {
    //LLVM_DEBUG(dbgs() << "skipping some non-innermost\n");
    LLVM_DEBUG(dbgs() << "  skipping non-innermost\n");
    return Changed;
  }
  ++NumInnermostLoops;

  // Skip loops with too many basic blocks
  if (Loop->getNumBlocks() > 8) {
    ++NumInnerLoopsTooManyBlocks;
    LLVM_DEBUG(dbgs() << "  skipping: Too many blocks (" << Loop->getNumBlocks() << " > 8)\n");
    return Changed;
  }

  // Count instructions in the loop and check for internal branches
  // We want to skip loops that have branches within the loop body
  // (excluding the terminator back-edge branch)
  MachineInstr *BackedgeBranch = findLoopBackedgeBranch(Loop);
  unsigned LoopCount = 0;
  SmallVector<MachineInstr *, 4> InternalBranches;
#ifndef LOOPUNROLLASM_ALLOW_ATOMIC_UNROLL
  bool hasAtomicOps = false;
#endif

  // Calculate a precise LoopCount based on the loop size (in # fetch slots)
  // TODO: this should eventually interface with TTI
  MachineInstr *PrevInst = nullptr;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      // Skip debug instructions and pseudo instructions
      if (MI.isDebugInstr() || MI.isPseudo())
        continue;

      StringRef OpcodeName = TII->getName(MI.getOpcode());
      if (OpcodeName == "FCSEL") {
        ++NumInnerLoopsHasFcsel;
        LLVM_DEBUG(dbgs() << "  skipping: FCSEL instruction\n");
        return Changed;
      }

      ++LoopCount;
      if (PrevInst && MI.isBranch() && isCompareBranchFusion(PrevInst, &MI, TII))
        // Compare-Branch pair get fused into one operation
        --LoopCount;
      else if (isPostIndexMemOp(MI))
        // Post-index operations translate into 2 operations
        ++LoopCount;
      if (isLoadStorePair(MI))
        // Load/store pair instructions translate into 2 operations (orthogonal to Post-Index)
        ++LoopCount;

      // Lambda to check if OpcodeName starts with any of the given mnemonics
      auto startsWithAny = [&OpcodeName](std::initializer_list<StringRef> mnemonics) {
        return llvm::any_of(mnemonics, [&OpcodeName](StringRef mnemonic) {
          return OpcodeName.starts_with(mnemonic);
        });
      };

      // Check for WFE (Wait For Event) instruction
      if (OpcodeName == "WFE") {
        // WFE instruction has additional overhead
        ++LoopCount;
        LLVM_DEBUG(dbgs() << "    Observed WFE instruction\n");
      }
      // Check for Pointer Authentication (AUT*) instructions
      else if (startsWithAny({"AUT"})) {
        // Pointer Authentication instructions have additional overhead
        ++LoopCount;
        LLVM_DEBUG(dbgs() << "    Observed Pointer Authentication instruction: " << OpcodeName << "\n");
      }
      // Atomic instructions are multiple operations
      // LDADD* variants (LDADD, LDADDA, LDADDAL, LDADDL, etc.)
      // CAS* variants (CAS, CASA, CASAL, CASL, CASB, CASH, etc.)
      else if (startsWithAny({"LDADD", "CAS"})) {
        LoopCount += 3;
        LLVM_DEBUG(dbgs() << "    Observed atomic instruction: " << OpcodeName << "\n");
      }
      // Check for integer MADD (Multiply-Add) instructions
      // MADD, MADDWrrr, MADDXrrr, SMADDL, UMADDL (but not FMADD for floating-point)
      else if (startsWithAny({"MADD", "SMADD", "UMADD"}) && !OpcodeName.starts_with("FMADD")) {
        // Integer multiply-add has additional overhead
        ++LoopCount;
        LLVM_DEBUG(dbgs() << "    Observed integer MADD instruction: " << OpcodeName << "\n");
      }
      // Check for Synchronization Barrier instructions
      // DMB (Data Memory Barrier), DSB (Data Synchronization Barrier), ISB (Instruction Synchronization Barrier)
      else if (startsWithAny({"DMB", "DSB", "ISB"})) {
        // Synchronization barriers have very significant overhead (4 additional cycles)
        LoopCount += 4;
        LLVM_DEBUG(dbgs() << "    Observed synchronization barrier instruction: " << OpcodeName << "\n");
      }

      // We want to exclude loops with any control flow changing instructions
      // (branches, calls, returns) except for the terminator
      if ((MI.isBranch() || MI.isCall() || MI.isReturn()) &&
          !MI.isTerminator()) {
        InternalBranches.push_back(&MI);
      }

#ifndef LOOPUNROLLASM_ALLOW_ATOMIC_UNROLL
      // Check for atomic operations
      // We want to avoid unrolling loops with atomic operations,
      // especially exclusive load/store operations (ldxr/stxr on ARM)
      if (MI.hasOrderedMemoryRef() || MI.mayLoadOrStore()) {
        // Check if this is an atomic operation by looking at the opcode name
        // or memory operand flags
        StringRef OpcodeName = TII->getName(MI.getOpcode());
        // Check for exclusive operations (ARM specific)
        if (OpcodeName.contains_insensitive("ldxr") ||
            OpcodeName.contains_insensitive("stxr") ||
            OpcodeName.contains_insensitive("ldaxr") ||
            OpcodeName.contains_insensitive("stlxr") ||
            OpcodeName.contains_insensitive("cas") ||
            OpcodeName.contains_insensitive("swp") ||
            OpcodeName.contains_insensitive("ldadd") ||
            OpcodeName.contains_insensitive("ldclr") ||
            OpcodeName.contains_insensitive("ldeor") ||
            OpcodeName.contains_insensitive("ldset")) {
          hasAtomicOps = true;
          break;
        }

        // Also check memory operands for atomic flags
        for (MachineMemOperand *MMO : MI.memoperands()) {
          if (MMO->isAtomic()) {
            hasAtomicOps = true;
            break;
          }
        }
        if (hasAtomicOps)
          break;
      }
#endif // LOOPUNROLLASM_ALLOW_ATOMIC_UNROLL

      // Track previous instruction for fusion detection
      PrevInst = &MI;
    }
  }

  // Lambda to check if a branch instruction targets any block outside the loop
  auto branchTargetsExit = [&](const MachineInstr *MI) -> bool {
    for (const MachineOperand &MO : MI->operands())
      if (MO.isMBB() && !Loop->contains(MO.getMBB()))
        return true;
    return false;
  };

  // Count terminators across all loop blocks and check patterns
  SmallVector<MachineInstr *, 4> Terminators;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (auto I = MBB->rbegin(); I != MBB->rend(); ++I) {
      if (I->isTerminator()) {
        Terminators.push_back(&*I);
      } else {
        break; // Stop at first non-terminator
      }
    }
  }
  unsigned NumTerminators = Terminators.size();
  TerminatorPattern Pattern = (NumTerminators == 1 && (LoopUnrollASMEnable & 0x1)) ? One_Backedge : Nonsupported;

  // Traverse the backedge branch's block backwards to determine if loop ends with
  // unconditional branch somewhere outside the loop.
  MachineInstr *lastUnconditionalExitBranch = nullptr;
  if (MachineBasicBlock *BackedgeBlock = BackedgeBranch ? BackedgeBranch->getParent() : nullptr) {
    for (auto I = BackedgeBlock->rbegin(); I != BackedgeBlock->rend(); ++I) {
      MachineInstr &MI = *I;
      // Skip debug and pseudo instructions
      if (MI.isDebugInstr() || MI.isPseudo())
        continue;

      if (MI.isUnconditionalBranch() && branchTargetsExit(&MI)) {
        lastUnconditionalExitBranch = &MI;
        LLVM_DEBUG(dbgs() << "  Loop ends with unconditional exit branch\n");
      }
      break; // Found the first valid instruction before backedge
    }
  }

  // Check if this is a loop with multiple terminators where all non-backedge terminators target exit
  LLVM_DEBUG({dbgs() << "  info: NumTerminators=" << NumTerminators
                     << " isLoopSimplifyForm=" << isLoopSimplifyForm(Loop)
                     << " endsUncondExitBranch=" << (lastUnconditionalExitBranch != nullptr)
                     << " InternalBranches=" << InternalBranches.size()
                     << "\n";
    for (auto T: Terminators)
      dbgs() << "    info: cond=" << T->isConditionalBranch() <<
                " branchTargetsExit=" << branchTargetsExit(T) <<
                " backedge=" << (T == BackedgeBranch) << " " << *T;
  });

  MachineBasicBlock *Latch = Loop->getLoopLatch();
  if (Pattern == Nonsupported && Latch) {
    MachineBasicBlock::iterator LastIter = Latch->getLastNonDebugInstr();
    if (LastIter == Latch->end()) {
      ++NumInnerLoopsInvalid;
      LLVM_DEBUG(dbgs() << "  skipping: Invalid loop: No non-debug instructions\n");
      return Changed;
    }

    // Check for backedge conditional followed by unconditional exit in the Latch block
    // TODO: tweak the loop_with_cond_uncond_pattern test to be detected here (avoid Branch Folding optimization)
    MachineInstr *Last = &*LastIter;
    MachineBasicBlock::iterator FirstTermIter = Latch->getFirstTerminator();
    if (FirstTermIter != LastIter) {
      auto SecondTermIter = std::next(FirstTermIter);
      if (SecondTermIter == LastIter &&
          FirstTermIter != Latch->end() &&
          FirstTermIter->isConditionalBranch() &&
          Last->isUnconditionalBranch()) {
        // Pattern 1: conditional branch followed by unconditional branch
        if (LoopUnrollASMEnable & 0x2) {
          Pattern = Two_Backedge_Uncond;
          LLVM_DEBUG(dbgs() << "  Detected Two_Backedge_Uncond pattern\n");
        }
        if (Last != lastUnconditionalExitBranch) {
          ++NumInnerLoops_InvalidUncondExit;
          LLVM_DEBUG(dbgs() << "    Last != lastUnconditionalExitBranc\n");
        }
      }
    }

    // Check if the last instruction is actually a branch
    // In some cases (e.g., when there are no terminators), Last may not be a branch
    if (Pattern == Nonsupported && !Last->isBranch()) {
      ++NumInnerLoops_LastInstNotBranch;
      LLVM_DEBUG(dbgs() << "  warning: Invalid loop: Last instruction is not a branch\n");
      //return Changed;
    }
  }
  if (NumTerminators > 1 && Pattern == Nonsupported) {
    // TODO: this may replace the more complex Latch based logic above
    if (Pattern == Nonsupported && lastUnconditionalExitBranch && NumTerminators == 2 && BackedgeBranch) {
      if (LoopUnrollASMEnable & 0x2) {
        Pattern = Two_Backedge_Uncond;
        LLVM_DEBUG(dbgs() << "  Detected Two_Backedge_Uncond pattern; fixup\n");
      }
    }

    // Check if all terminators except BackedgeBranch target the exit
    // can now override a previous Two_Backedge_Uncond pattern
    else if (BackedgeBranch) {
      bool allOthersTargetExit = true;
      for (auto T : Terminators) {
        if ((T != BackedgeBranch && !branchTargetsExit(T))) {
          allOthersTargetExit = false;
          break;
        }
      }
      if (allOthersTargetExit && (LoopUnrollASMEnable & 0x4)) {
        Pattern = Multi_CondExit_Backedge;
        LLVM_DEBUG(dbgs() << "  Detected Multi_CondExit_Backedge pattern ("
                          << NumTerminators << " terminators: exit branches + backedge)\n");
      }
    }

    if (Pattern == Nonsupported) {
      if (NumTerminators == 2)
        ++NumInnerLoopsMultipleTerminators2;
      else if (NumTerminators == 3)
        ++NumInnerLoopsMultipleTerminators3;
      else if (NumTerminators >= 4)
        ++NumInnerLoopsMultipleTerminators4Plus;
      LLVM_DEBUG(dbgs() << "  skipping: Multiple terminators (" << NumTerminators << ", not accepted pattern)\n");
      return Changed;
    }
  } // NumTerminators > 1

  if (!BackedgeBranch) {
    ++NumInnerLoopsBranchConditionalNoBackedge;
    LLVM_DEBUG(dbgs() << "  skipping: Loop has no backedge branch\n");
    return Changed;
  }

  SmallVector<MachineBasicBlock *, 4> Latches;
  Loop->getLoopLatches(Latches);
  assert(Latches.size() == 1 && "Loop must have single latch");

  // Classify the branch type and skip if not suitable for unrolling
  if (BackedgeBranch->isUnconditionalBranch()) {
    ++NumInnerLoopsBranchUnconditional;
    LLVM_DEBUG(dbgs() << "  skipping: Unconditional branch\n");
    // TODO: we should be able to unroll simple cases like:
    // Loop:  inst1
    //        b.eq Exit
    //        b Loop
    //  Exit:
    return Changed;
  }

  if (BackedgeBranch->isIndirectBranch()) {
    ++NumInnerLoopsBranchIndirect;
    LLVM_DEBUG(dbgs() << "  skipping: Indirect branch\n");
    return Changed;
  }
  assert(BackedgeBranch->isConditionalBranch() && "Loop with non-conditional backedge branch!");

  // Check if this is a single basic block loop (Head == Latch)
  // Exception: Multi_CondExit_Backedge pattern can have multiple blocks
  if (Pattern != Multi_CondExit_Backedge && (!Header || !Latch || Header != Latch)) {
    ++NumInnerLoopsNotSingleBB;
    LLVM_DEBUG(dbgs() << "  skipping: Not single BB (Header != Latch)\n");
    return Changed;
  }

#ifndef LOOPUNROLLASM_ALLOW_ATOMIC_UNROLL
  if (hasAtomicOps) {
    ++NumInnerLoopsHasAtomicOps;
    LLVM_DEBUG(dbgs() << "  skipping: Has atomic ops\n");
    return Changed;
  }
#endif

  // Find all exit blocks (successors that are outside the loop)
  SmallVector<MachineBasicBlock*, 4> ExitBlocks;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (!Loop->contains(Succ) &&
          std::find(ExitBlocks.begin(), ExitBlocks.end(), Succ) == ExitBlocks.end()) {
        ExitBlocks.push_back(Succ);
      }
    }
  }
  if (ExitBlocks.empty()) {
    ++NumInnerLoopsInvalid;
    LLVM_DEBUG(dbgs() << "  skipping: Invalid loop: no exit blocks\n");
    return Changed;
  }
  MachineBasicBlock *ExitBlock = ExitBlocks[0];

  if (!InternalBranches.empty()) {
    ++NumInnerLoopsHasInternalBranch;
    LLVM_DEBUG(dbgs() << "  skipping: Has internal branch\n");
    return Changed;
  }

  if (LoopCount >= LoopUnrollASMMaxInsts) {
    ++NumInnerLoopsTooManyInsts;
    LLVM_DEBUG(dbgs() << "  skipping: Too many instructions (" << LoopCount << " >= " << LoopUnrollASMMaxInsts << ")\n");
    return Changed;
  }

  // an unconditional exit branch that ends a loop won't be unrolled
  unsigned AdjustedLoopCount = lastUnconditionalExitBranch ? LoopCount - 1 : LoopCount;

  SmallVector<MachineOperand, 4> EmptyCond;
  BackedgeInfo BackedgeInfoObj(BackedgeBranch, Pattern, ExitBlock, EmptyCond, EmptyCond,
                                lastUnconditionalExitBranch != nullptr);
  return processTightLoop(Loop, MF, Header, AdjustedLoopCount, BackedgeInfoObj);
}

void LoopUnrollASM::debugPrintLoopInfo(MachineFunction &MF,
                                       const MachineLoop *Loop,
                                       StringRef Prefix,
                                       MachineBasicBlock *ExitBlock) {
  MachineBasicBlock *Header = Loop->getHeader();
  // Get debug location information
  DebugLoc DL;
  for (MachineInstr &MI : *Header) {
    if (!MI.isDebugInstr() && MI.getDebugLoc()) {
      DL = MI.getDebugLoc();
      break;
    }
  }

  LLVM_DEBUG({
    dbgs() << Prefix << " loop in function " << MF.getName() << "\n";
    if (DL) {
      dbgs() << "  Source location: ";
      if (DL.getLine() != 0 && DL.getScope()) {
        if (auto *Scope = dyn_cast<DIScope>(DL.getScope())) {
          dbgs() << Scope->getFilename() << ":" << DL.getLine();
          if (DL.getCol() != 0)
            dbgs() << ":" << DL.getCol();
        }
      }
      dbgs() << "\n";
    }

    // Helper lambda to print block name (use number if symbol name is empty)
    auto printBlockName = [](const MachineBasicBlock *MBB) {
      if (MCSymbol *Sym = MBB->getSymbol()) {
        StringRef Name = Sym->getName();
        if (!Name.empty()) {
          dbgs() << Name;
          return;
        }
      }
      dbgs() << "BB#" << MBB->getNumber();
    };

    dbgs() << "  Header=";
    printBlockName(Header);
    dbgs() << ", BasicBlocks in layout order: ";
    for (MachineBasicBlock &MBB : MF)
      if (Loop->contains(&MBB)) {
        printBlockName(&MBB);
        dbgs() << " , ";
      }
    if (ExitBlock) {
      dbgs() << "  ExitBlock: ";
      printBlockName(ExitBlock);
    }
    dbgs() << "\n";
  });
}

// processTightLoop() targets certain loops that meet these conditions:
// - Are Inner-most loops
// - Have less than N instructions (default 46, configurable via
// -loop-unroll-asm-max-insts)
// - Are single basic-block loops (Head == Latch)
// - Have a conditional branch as the backedge
// - Loop body has no control flow instructions (branches/calls/returns)
// - Loop body has no atomic operations (ldxr/stxr, atomicrmw, etc.)
bool LoopUnrollASM::processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                                     MachineBasicBlock *Header,
                                     unsigned LoopCount,
                                     BackedgeInfo &Backedge) {
  ++NumLoopsDetected;

  // TODO: get MachineWidth from TTI / SchedModel
  const unsigned MachineWidth = 10;
  const unsigned LoopCycles =
      (LoopCount + MachineWidth - 1) / MachineWidth; // round-up int divide

  unsigned Bubbles = calculateBubbles(LoopCount);
  // Hanlde the case if the loop would possibly induce +20% Frontend Bound
  if (Bubbles / float(MachineWidth * LoopCycles) > LoopUnrollASMFetchBubblesThreshold) {
    // First, we need to analyze the loop branch to see if we can invert it
    // Analyze the original branch to extract condition
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    if (TII->analyzeBranch(*Backedge.Branch->getParent(), TBB, FBB, Backedge.Cond)) {
      LLVM_DEBUG(dbgs() << "  Unable to analyze branch for unrolling\n");
      ++NumInnerLoopsBranchPrepFailure;
      return false;
    }

    // Fall-through could vary in case of conditional branch with fall-through (FBB==null)
    if (TBB && !FBB) {
      MachineBasicBlock *FallthroughBB = Backedge.Branch->getParent()->getNextNode();
      if (FallthroughBB && FallthroughBB != Backedge.ExitBlock) {
        LLVM_DEBUG(dbgs() << "  saw a different fallthrough vs exit blocks\n");
        Backedge.ExitBlock = FallthroughBB;
      }
    }

    // Try to invert the condition
    Backedge.InvertedCond = Backedge.Cond;
    if (TII->reverseBranchCondition(Backedge.InvertedCond)) {
      LLVM_DEBUG(dbgs() << "  Unable to invert branch condition, skipping unrolling\n");
      ++NumInnerLoopsBranchPrepFailure;
      return false;
    }

    LLVM_DEBUG({debugPrintLoopInfo(MF, Loop, "Found qualifying", Backedge.ExitBlock);
      dbgs() << "  with Loop count: " << LoopCount << "\n";});
    unsigned UC = findBestUnrollCount(LoopCount, Bubbles, MachineWidth);
    duplicateLoopBody(Loop, UC, Backedge);
    NumAddedInsts += (UC - 1) * LoopCount;
    ++NumLoopsUnrolled;
    if (Backedge.Pattern == Two_Backedge_Uncond) {
      ++NumLoopsUnrolledCondUncond;
    } else if (Backedge.Pattern == Multi_CondExit_Backedge) {
      ++NumLoopsUnrolledMultiCondExit;
    }
    return true;
  }
  LLVM_DEBUG(dbgs() << "  bubbles are good\n");

  ++NumLoopsNotEnoughBubbles;
  return false;
}

unsigned LoopUnrollASM::findBestUnrollCount(unsigned LoopCount,
                                            unsigned Bubbles,
                                            unsigned MachineWidth) {
  unsigned BestUC = 2; // Start with the default unroll factor of 2
  unsigned UC = 2;

  while (true) {
    // Calculate the new loop count after unrolling
    unsigned NewLoopCount = LoopCount * UC;

    // Stop if the new loop count is too large
    if (NewLoopCount > 12 * MachineWidth)
      break;

    // Calculate new bubbles for this unroll count
    unsigned NewBubbles = calculateBubbles(NewLoopCount);
    // Update best unroll count if we found fewer bubbles
    if (NewBubbles < Bubbles) {
      BestUC = UC;
      Bubbles = NewBubbles; // Update Bubbles for comparison in next iteration
    }

    // Stop if we've eliminated all bubbles
    if (NewBubbles == 0)
      break;

    ++UC;
  }

  return BestUC;
}

void LoopUnrollASM::duplicateLoopBody(MachineLoop *Loop,
                                      unsigned UnrollFactor,
                                      const BackedgeInfo &Backedge) {
  assert (UnrollFactor > 1);

  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineBasicBlock *Header = Loop->getHeader();
  MachineFunction *MF = Header->getParent();
  MachineBasicBlock *BackedgeBlock = Backedge.Branch->getParent();

  // Collect all non-terminator instructions to duplicate from all loop blocks
  // Use a map to maintain per-block instruction lists for multi-block loops
  DenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 16>> InstsToClone;

  // Collect loop blocks in their layout order (not Loop->blocks() order)
  SmallVector<MachineBasicBlock *, 4> LoopBlocksInOrder;
  for (MachineBasicBlock &MBB : *MF) {
    if (Loop->contains(&MBB)) {
      LoopBlocksInOrder.push_back(&MBB);
    }
  }

  for (MachineBasicBlock *MBB : LoopBlocksInOrder) {
    SmallVector<MachineInstr *, 16> &BlockInsts = InstsToClone[MBB];
    for (MachineInstr &MI : *MBB) {
      // Skip PHI nodes and debug instructions
      if (MI.isPHI() || MI.isDebugInstr())
        continue;

      // For the backedge block, skip terminators (we'll insert new ones)
      // For other blocks, include terminators (they need to be cloned)
      if (MI.isTerminator() && MBB == BackedgeBlock)
        continue;

      BlockInsts.push_back(&MI);
    }
  }

  // Use the branch analysis results passed in from the parent function

  unsigned TotalInsts = 0;
  for (const auto &Entry : InstsToClone)
    TotalInsts += Entry.second.size();
  TotalInsts += 1; // for Backedge terminator
  LLVM_DEBUG(dbgs() << "  Original loop has " << TotalInsts << " instructions spanning " << Loop->getNumBlocks() << " blocks.\n");

  // We need to create new basic blocks for proper control flow
  // The structure will be:
  // Header -> Body1 -> Body2 -> ... -> BodyN -> Header (loop back)
  //    |        |        |              |
  //    v        v        v              v
  //  Exit    Exit      Exit           Exit

  // Create UnrollFactor-1 new basic blocks for each original loop block
  // (we reuse the original blocks for the first iteration)
  SmallVector<SmallVector<MachineBasicBlock *, 4>, 4> NewBlocks(UnrollFactor - 1);
  MachineBasicBlock *PrevBlock = LoopBlocksInOrder.back(); // Last block in original loop order

  for (unsigned i = 1; i < UnrollFactor; ++i) {
    // Create a new block for each original loop block in layout order
    for (MachineBasicBlock *OrigMBB : LoopBlocksInOrder) {
      MachineBasicBlock *NewBB = MF->CreateMachineBasicBlock();
      MF->insert(++MachineFunction::iterator(PrevBlock), NewBB);
      // Copy live-in registers from the original block
      for (const auto &LI : OrigMBB->liveins())
        NewBB->addLiveIn(LI.PhysReg);
      NewBlocks[i - 1].push_back(NewBB);
      PrevBlock = NewBB;
    }
  }
  assert(!NewBlocks.empty() && "NewBlocks should not be empty when UnrollFactor > 1");

  // Remove the original branch from the backedge block - we'll add new ones
  TII->removeBranch(*BackedgeBlock);

  // Now set up each iteration with its body blocks and appropriate branches
  for (unsigned i = 0; i < UnrollFactor; ++i) {
    unsigned BlockIdx = 0;
    // For each original loop block in layout order, process the corresponding block in this iteration
    for (MachineBasicBlock *OrigMBB : LoopBlocksInOrder) {
      MachineBasicBlock *CurrentBlock = (i == 0) ? OrigMBB : NewBlocks[i - 1][BlockIdx];
      if  (i>0) {
        // Clone body instructions into the new block
        for (MachineInstr *MI : InstsToClone[OrigMBB]) {
          MachineInstr *ClonedMI = MF->CloneMachineInstr(MI);
          CurrentBlock->push_back(ClonedMI);
        }
      }

      // Only insert branches for the backedge block
      // For Multi_CondExit_Backedge pattern, other blocks keep their original terminators
      if (OrigMBB == BackedgeBlock) {
        // Insert appropriate branch for the backedge block in each iteration
        if (i < UnrollFactor - 1) {
          // For all but the last iteration, use the inverted conditional branch
          // Branch to exit on condition, or continue to next iteration's first block
          assert(CurrentBlock->isLayoutSuccessor(NewBlocks[i][0]) && "NewBlock not Successor when duplicating!");
          TII->insertBranch(*CurrentBlock, Backedge.ExitBlock, nullptr,
                            Backedge.InvertedCond, Backedge.Branch->getDebugLoc());
        } else {
          // For the last iteration, branch back to header with original condition
          // (loop back when we should continue, exit otherwise)
          TII->insertBranch(*CurrentBlock, Header,
                            CurrentBlock->isLayoutSuccessor(Backedge.ExitBlock) ? nullptr : Backedge.ExitBlock,
                            Backedge.Cond, Backedge.Branch->getDebugLoc());
        }
      }

      BlockIdx++;
    } // for LoopBlocksInOrder
  }

  // Update the CFG - fix successor relationships
  // First update the backedge block's successors (last block of first iteration)
  // Backedge block now branches to either first block of second iteration or exit
  // Remove self-loop to header if it exists (only for single-block loops where backedge is in header)
  if (BackedgeBlock->isSuccessor(Header))
    BackedgeBlock->removeSuccessor(Header);
  if (!BackedgeBlock->isSuccessor(NewBlocks[0][0]))
    BackedgeBlock->addSuccessor(NewBlocks[0][0]);
  // ExitBlock successor should already be there

  // For Multi_CondExit_Backedge pattern, update successors for original non-backedge blocks
  // These blocks keep their terminators but need CFG successors updated to point to
  // duplicated blocks in the first unrolled iteration
  if (Backedge.Pattern == Multi_CondExit_Backedge) {
    for (unsigned j = 0; j < LoopBlocksInOrder.size(); ++j) {
      MachineBasicBlock *OrigMBB = LoopBlocksInOrder[j];

      // Skip the backedge block (already handled above)
      if (OrigMBB == BackedgeBlock)
        continue;

      // For non-backedge blocks in the original iteration, update their successors
      // to point to the corresponding blocks in the first duplicated iteration
      SmallVector<MachineBasicBlock *, 4> OldSuccessors(OrigMBB->successors().begin(),
                                                         OrigMBB->successors().end());
      for (MachineBasicBlock *OrigSucc : OldSuccessors) {
        if (Loop->contains(OrigSucc)) {
          // Internal successor - need to map to the corresponding block in first duplicated iteration
          auto It = llvm::find(LoopBlocksInOrder, OrigSucc);
          if (It != LoopBlocksInOrder.end()) {
            unsigned SuccIdx = std::distance(LoopBlocksInOrder.begin(), It);
            MachineBasicBlock *NewSucc = NewBlocks[0][SuccIdx];
            // Remove old internal successor and add new one
            OrigMBB->removeSuccessor(OrigSucc);
            if (!OrigMBB->isSuccessor(NewSucc))
              OrigMBB->addSuccessor(NewSucc);
          }
        }
        // External successors (exit blocks) are kept as-is
      }
    }
  }

  // Now add successors for the new iteration blocks
  // Each iteration has multiple blocks in the same order as original loop
  for (unsigned i = 0; i < NewBlocks.size(); ++i) {
    for (unsigned j = 0; j < NewBlocks[i].size(); ++j) {
      MachineBasicBlock *BB = NewBlocks[i][j];
      MachineBasicBlock *OrigMBB = LoopBlocksInOrder[j];
      bool isLastBlockInIteration = (j == NewBlocks[i].size() - 1);

      if (isLastBlockInIteration) {
        // Last block in iteration (backedge block) can exit
        if (!BB->isSuccessor(Backedge.ExitBlock))
          BB->addSuccessor(Backedge.ExitBlock);

        // And can continue to first block of next iteration or loop back to header
        if (i < NewBlocks.size() - 1) {
          if (!BB->isSuccessor(NewBlocks[i + 1][0]))
            BB->addSuccessor(NewBlocks[i + 1][0]);
        } else {
          // Last iteration's last block loops back to header
          if (!BB->isSuccessor(Header))
            BB->addSuccessor(Header);
        }
      } else {
        // Non-last blocks within an iteration need their successors set up
        // These blocks may have conditional exits and internal successors
        for (MachineBasicBlock *OrigSucc : OrigMBB->successors()) {
          if (!Loop->contains(OrigSucc)) {
            // External successor (exit block) - keep as is
            if (!BB->isSuccessor(OrigSucc))
              BB->addSuccessor(OrigSucc);
          } else {
            // Internal successor - need to map to the corresponding block in this iteration
            // Find the index of the successor in the loop blocks
            auto It = llvm::find(LoopBlocksInOrder, OrigSucc);
            if (It != LoopBlocksInOrder.end()) {
              unsigned SuccIdx = std::distance(LoopBlocksInOrder.begin(), It);
              MachineBasicBlock *NewSucc = NewBlocks[i][SuccIdx];
              if (!BB->isSuccessor(NewSucc))
                BB->addSuccessor(NewSucc);
            }
          }
        }
      }
    }
  }

  // Add all newly created blocks to the Loop
  for (const auto &IterBlocks : NewBlocks) {
    for (MachineBasicBlock *BB : IterBlocks) {
      Loop->addBasicBlockToLoop(BB, MLI);
    }
  }

  LLVM_DEBUG(dbgs() << "  Duplicated loop body with unroll factor " << UnrollFactor << "\n");

  // Verify instruction count matches expectations
  unsigned NewLoopInstCount = 0;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (MI.isPHI() || MI.isDebugInstr())
        continue;
      ++NewLoopInstCount;
    }
  }

  unsigned ExpectedInstCount = UnrollFactor * TotalInsts;
  if (NewLoopInstCount != ExpectedInstCount && NewLoopInstCount != (ExpectedInstCount+1)) {
    dbgs() << "ERROR: Instruction count mismatch after loop unrolling!\n";
    dbgs() << "  Pattern: " << Backedge.Pattern << " EndsWithUncondExit=" << Backedge.EndsWithUncondExit << "\n";
    dbgs() << "  Expected: " << ExpectedInstCount
           << " (UnrollFactor=" << UnrollFactor << " * TotalInsts=" << TotalInsts << ")\n";
    dbgs() << "  Actual: " << NewLoopInstCount << "\n";

    debugPrintLoopInfo(*MF, Loop, " ", Backedge.ExitBlock);

    dbgs() << "  Instruction counts per block of original loop:\n";
    for (MachineBasicBlock *MBB : LoopBlocksInOrder) {
      unsigned BlockInstCount = 0;
      dbgs() << "    Block " << MBB->getNumber() << ":\n";
      for (MachineInstr &MI : *MBB) {
        if (MI.isPHI() || MI.isDebugInstr())
          continue;
        dbgs() << "      " << MI;
        ++BlockInstCount;
      }
      dbgs() << "      Total: " << BlockInstCount << " instructions\n";
    }

    assert(false && "Instruction count mismatch after loop unrolling");
  }
}

FunctionPass *llvm::createLoopUnrollASMPass() {
  return new LoopUnrollASM();
}
