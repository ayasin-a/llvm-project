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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-asm"

static cl::opt<bool> LoopUnrollASMAlign(
    "loop-unroll-asm-align",
    cl::desc("Enable alignment-specific analysis"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> LoopUnrollASMAlignAll(
    "loop-unroll-asm-align-all",
    cl::desc("Analyze all basic blocks for alignment issues (not just innermost loops)"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> LoopUnrollASMAlignByDirective(
    "loop-unroll-asm-align-by-directive",
    cl::desc("Enable alignment optimization for FCMP-FCSEL straddling"),
    cl::init(true), cl::Hidden);

static cl::opt<int> LoopUnrollASMAlignThreshold(
    "loop-unroll-asm-align-threshold",
    cl::desc("Threshold for 64-byte alignment of unrolled loops (0-15: bubble threshold, -1: disable alignment)"),
    cl::init(-1), cl::Hidden,
    cl::callback([](const int &Val) {
      if (Val < -1 || Val > 15)report_fatal_error("loop-unroll-asm-align-threshold must be -1 or in range [0, 15]");
    }));

static cl::opt<unsigned> LoopUnrollASMDebug(
    "loop-unroll-asm-debug",
    cl::desc("Debug level for loop unroll asm pass"),
    cl::init(2), cl::Hidden);

// **Debug Level Hierarchy**:
//    - Level 0: Warning messages (always shown, e.g., "Warning: No backedge blocks found")
//    - Level 1: Per-function high-level (e.g., "Analyzing function: X")
//    - Level 2: Per-function low-level (detailed function information)
//    - Level 3: Per-loop high-level (e.g., "Examining loop", "Found qualifying loop")
//    - Level 4: Per-loop low-level (e.g., "findBestUnrollCount", loop skipping reasons)
//    - Level 5: Per-BB high-level (e.g., basic block processing, branch patterns)
//    - Level 6: Per-BB low-level (e.g., detailed BB analysis, crossing detection details)
//    - Level 7: Per-instruction high-level (e.g., fusion detection, instruction classification)
//    - Level 8: Per-instruction low-level (e.g., detailed offset calculations, per-instruction details)
//    - Level 10: detailed debug.
// DBG macro that takes level as first parameter

static cl::opt<unsigned> LoopUnrollASMEnable(
    "loop-unroll-asm-enable",
    cl::desc("Bitmask to enable each pattern in LoopUnrollASM "
             "(bit 0: One_Backedge, bit 1: Two_Backedge_Uncond, bit 2: Multi_CondExit_Backedge, bit 3: Simple_SubLoop)"),
    cl::init(0xb), cl::Hidden);

static cl::opt<float> LoopUnrollASMFetchBubblesThreshold(
    "loop-unroll-asm-fetch-bubbles-threshold",
    cl::desc("Threshold for fetch bubbles ratio to trigger loop unrolling"),
    cl::init(0.20f), cl::Hidden);

static cl::opt<unsigned> LoopUnrollASMMaxBlocks(
    "loop-unroll-asm-max-blocks",
    cl::desc("Maximum number of basic blocks in a loop for LoopUnrollASM to process"),
    cl::init(8), cl::Hidden);

static cl::opt<unsigned> LoopUnrollASMMaxOps(
    "loop-unroll-asm-max-ops",
    cl::desc("Maximum number of operations in a loop for LoopUnrollASM to process"),
    cl::init(46), cl::Hidden);

static cl::opt<unsigned> LoopUnrollASMMinBubbles(
    "loop-unroll-asm-min-bubbles",
    cl::desc("Minimum number of bubbles to continue searching for better unroll count"),
    cl::init(3), cl::Hidden,
    cl::callback([](const unsigned &Val) {
      if (Val == 0) report_fatal_error("loop-unroll-asm-min-bubbles must be at least 1");
    }));

static cl::opt<unsigned> LoopUnrollASMSkip(
    "loop-unroll-asm-skip",
    cl::desc("Bitmask to control DO_SKIP heuristics "
             "(bit 0: HasMadd, bit 1: HasVectorByElement, bit 2: HasRsqrt, bit 3: HasCall)"),
    cl::init(0x1), cl::Hidden);

#define DBG(level, ...) do { \
  if (level <= LoopUnrollASMDebug) { \
    LLVM_DEBUG(__VA_ARGS__); \
  } \
} while(0)

// DBG_OBSERVED macro for instruction observations - calls DBG with level 7 and adds "    Observed " prefix
#define DBG_OBSERVED(str, opcode) DBG(7, dbgs() << "    Observed " << str << ": " << opcode << "\n");

// DBG_SKIP macro for simple skipping messages - calls DBG with level 3 and adds "  skipping: " prefix
#define DBG_SKIP(str) DBG(3, dbgs() << "  skipping: " << str << "\n");

// Helper macros to detect if optional parameter is provided
#define GET_MACRO(_1, _2, NAME, ...) NAME
#define DO_SKIP_1(suffix) { \
  ++NumInnerLoops##suffix; \
  DBG_SKIP(#suffix); \
  return Changed; \
}
#define DO_SKIP_2(suffix, msg) { \
  ++NumInnerLoops##suffix; \
  DBG(3, dbgs() << "  skipping: " << #suffix << " " << msg << "\n"); \
  return Changed; \
}

// DO_SKIP macro for complete skip pattern - increments statistic, logs skip message, and returns Changed
// Optional second parameter for additional details concatenated to the skip message
#define DO_SKIP(...) GET_MACRO(__VA_ARGS__, DO_SKIP_2, DO_SKIP_1, )(__VA_ARGS__)

// DO_SKIP_HELPER variant that returns std::nullopt for use in helper functions
#define DO_SKIP_HELPER(suffix) { \
  ++NumInnerLoops##suffix; \
  DBG_SKIP(#suffix); \
  return std::nullopt; \
}

// totals
STATISTIC(NumLoopsDetected, "Number of tight loops detected by LoopUnrollASM");
STATISTIC(NumLoopsUnrolled, "Number of tight loops unrolled by LoopUnrollASM");
STATISTIC(NumLoopsUnrolledCondUncond, "Number of tight loops unrolled with cond+uncond terminators");
STATISTIC(NumLoopsUnrolledMultiCondExit, "Number of tight loops unrolled with multi-cond-exit+backedge pattern");
STATISTIC(NumLoopsUnrolledSimpleSubLoop, "Number of tight loops unrolled with simple sub-loop pattern");
STATISTIC(NumInnermostLoopsAligned, "Number of innermost loops aligned to 64 bytes");
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
STATISTIC(NumInnerLoopsHasIndirect, "Number of inner loops skipped (indirect branch/call)");
STATISTIC(NumInnerLoopsBranchConditionalNoBackedge, "Number of inner loops skipped (conditional branch without backedge)");
STATISTIC(NumInnerLoopsHasAtomicOps, "Number of inner loops skipped (has atomic ops)");
STATISTIC(NumInnerLoopsHasInternalBranch, "Number of inner loops skipped (has internal branch)");
STATISTIC(NumInnerLoopsHasCall, "Number of inner loops skipped (has direct CALL omst)");
STATISTIC(NumInnerLoopsHasMadd, "Number of inner loops skipped (has MADD inst)");
STATISTIC(NumInnerLoopsHasVectorByElement, "Number of inner loops skipped (has vector load/store by-element)");
STATISTIC(NumInnerLoopsHasRsqrt, "Number of inner loops skipped (has RQSRT inst)");
STATISTIC(NumInnerLoopsTooManyBlocks, "Number of inner loops skipped (too many blocks)");
STATISTIC(NumInnerLoopsTooManyInsts, "Number of inner loops skipped (too many instructions)");
STATISTIC(NumInnerLoopsBranchPrepFailure, "Number of loops skipped (branch analysis or condition inversion failed)");
// dev stats
STATISTIC(NumInnerLoops_BackedgeFallthruHeader, "Number of inner loops where Backedge branch fallsthrough into Loop Header");
STATISTIC(NumInnerLoops_HasFcmpFcsel, "Number of inner loops with FCMP-FCSEL pair");
STATISTIC(NumInnerLoops_InvalidUncondExit, "Number of inner loops ending with weird unconditional exit branch");
STATISTIC(NumInnerLoops_LastInstNotBranch, "Number of inner loops where last instruction is not a branch");
// alignment pass
STATISTIC(NumAlignMBBsSkippedNotInnermost, "Number of MBBs skipped (not in innermost loop)");
STATISTIC(NumAlignmentsSetForStraddle, "Number of alignments set to prevent FCMP+FCSEL cacheline crossings");
STATISTIC(NumInnermostLoopsAlign8, "Number of innermost loops with 8-byte alignment");
STATISTIC(NumInnermostLoopsAlign16, "Number of innermost loops with 16-byte alignment");
STATISTIC(NumInnermostLoopsAlign32, "Number of innermost loops with 32-byte alignment");
STATISTIC(NumInnermostLoopsAlign64, "Number of innermost loops with 64-byte alignment");
STATISTIC(NumInnermostLoopsAlignOther, "Number of innermost loops with other alignment");
STATISTIC(NumFunctionsAlign8, "Number of functions with 8-byte alignment");
STATISTIC(NumFunctionsAlign16, "Number of functions with 16-byte alignment");
STATISTIC(NumFunctionsAlign32, "Number of functions with 32-byte alignment");
STATISTIC(NumFunctionsAlign64, "Number of functions with 64-byte alignment");
STATISTIC(NumFunctionsAlignOther, "Number of functions with other alignment");

namespace {
enum TerminatorPattern {
  Nonsupported = 0,
  One_Backedge,
  Two_Backedge_Uncond,
  Multi_CondExit_Backedge,
  Simple_SubLoop,
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
  const MachineLoopInfo *MLI = nullptr;

  // TODO: get MachineWidth from TTI / SchedModel
  static constexpr unsigned MachineWidth = 10;

  // Static helper function to calculate bubbles
  static inline unsigned calculateBubbles(unsigned LoopCount, unsigned Size) {
    unsigned Remainder = LoopCount % Size;
    return Remainder ? Size - Remainder : 0;
  }

  bool isCompareBranchFusion(const MachineInstr *CompareInst, const MachineInstr *BranchInst);
  bool isPostIndexMemOp(const MachineInstr &MI);
  bool isLoadStorePair(const MachineInstr &MI);
  static bool isLoopSimplifyForm(const MachineLoop *Loop);
  static void debugPrintLoopInfo(const MachineLoop *Loop, StringRef Prefix,
                          const SmallVectorImpl<MachineBasicBlock *> &Blocks,
                          MachineBasicBlock *ExitBlock = nullptr);
  static MachineInstr* findLoopBackedgeBranch(const MachineLoop* Loop);
  bool processLoop(MachineLoop *Loop, MachineFunction &MF);
  bool processTightLoop(MachineLoop *Loop, MachineBasicBlock *Header, unsigned LoopCount,
                        BackedgeInfo &Backedge, unsigned LoopInsts,
                        const SmallVectorImpl<MachineBasicBlock *> &BlocksInOrder);
  unsigned findBestUnrollCount(unsigned LoopCount, unsigned Bubbles);
  void duplicateLoopBody(MachineLoop *Loop, unsigned UnrollFactor,
                         const BackedgeInfo &Backedge, unsigned NumInsts,
                         const SmallVectorImpl<MachineBasicBlock *> &BlocksInOrder);
  std::optional<unsigned> calculateLoopCount(const iterator_range<MachineLoop::block_iterator> Blocks,
                                           SmallVectorImpl<MachineInstr *> &InternalBranches);
  unsigned countInstructions(const iterator_range<MachineLoop::block_iterator> Blocks);
  bool areLoopBlocksContiguous(const MachineLoop *Loop);
  SmallVector<MachineBasicBlock *, 4> findSubLoopBlocks(const SmallVectorImpl<MachineBasicBlock *> &LoopBlocksInOrder,
                                                       MachineInstr *&BackwardBranch, MachineBasicBlock *&NextBlock);
  bool analyzeMachineInsts(MachineFunction &MF);
  bool isAtomicInstruction(const MachineInstr &MI);
};
} // end anonymous namespace

char LoopUnrollASM::ID = 0;
char &llvm::LoopUnrollASMID = LoopUnrollASM::ID;

INITIALIZE_PASS_BEGIN(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)

bool LoopUnrollASM::runOnMachineFunction(MachineFunction &MF) {
  if (!LoopUnrollASMEnable && !LoopUnrollASMAlign)
    return false;
  DBG(1, dbgs() << "Analyzing function: " << MF.getName() << "\n");

  TII = MF.getSubtarget().getInstrInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  bool Changed = false;

  // Process loops if enabled
  if (LoopUnrollASMEnable) {

    if (!MLI->empty()) {
      // Process all loops, starting with innermost ones
      for (MachineLoop *Loop : *MLI) {
        Changed |= processLoop(Loop, MF);
      }
    }
  }

  if (LoopUnrollASMAlign) {
    Changed |= analyzeMachineInsts(MF);
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
    DBG(0, dbgs() << "Warning: No backedge blocks found for loop\n");
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
      DBG(0, dbgs() << "Warning: Multiple backedge blocks, choosing block "
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
        DBG(5, dbgs() << "  backedge falls-through loop header\n");
        ++NumInnerLoops_BackedgeFallthruHeader;
      }
    }

    if (BranchesToHeader) {
      BackedgeBranch = &MI;
      break;
    }
  }

  DBG(0, if (!BackedgeBranch) dbgs() << "Warning: No branch found that targets loop header\n");

  return BackedgeBranch;
}

/// Check if two consecutive instructions form a Compare-Branch Fusion sequence
/// according to ARM CPU architecture specifications.
/// Returns true if fusion can occur, false otherwise.
bool LoopUnrollASM::isCompareBranchFusion(const MachineInstr *CompareInst,
                                          const MachineInstr *BranchInst) {
  if (!CompareInst || !BranchInst)
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
    DBG(7, dbgs() << "    Observed Compare-Branch Fusion (Case 1): "
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
      DBG(7, dbgs() << "    Observed Compare-Branch Fusion (Case 2): "
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
    DBG_OBSERVED("post-index instruction in", OpcodeName);
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
    DBG_OBSERVED("load/store pair instruction", OpcodeName);
  }

  return isPairOp;
}

/// Check if a machine instruction is an atomic operation.
bool LoopUnrollASM::isAtomicInstruction(const MachineInstr &MI) {
  if (!MI.hasOrderedMemoryRef() && !MI.mayLoadOrStore())
    return false;

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
    return true;
  }

  // Also check memory operands for atomic flags
  for (MachineMemOperand *MMO : MI.memoperands()) {
    if (MMO->isAtomic()) {
      return true;
    }
  }

  return false;
}

FunctionPass *llvm::createLoopUnrollASMPass() {
  return new LoopUnrollASM();
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

// Count instructions in the (sub)loop and determine internal branches
std::optional<unsigned> LoopUnrollASM::calculateLoopCount(const iterator_range<MachineLoop::block_iterator> Blocks,
                                                        SmallVectorImpl<MachineInstr *> &InternalBranches) {
  unsigned LoopCount = 0;

  // Calculate a precise LoopCount based on the loop size (in # fetch slots)
  // TODO: this should eventually interface with TTI
  MachineInstr *PrevInst = nullptr;
  for (MachineBasicBlock *MBB : Blocks) {
    for (MachineInstr &MI : *MBB) {
      // Skip debug instructions and pseudo instructions
      if (MI.isDebugInstr() || MI.isPseudo())
        continue;

      StringRef OpcodeName = TII->getName(MI.getOpcode());
      if (OpcodeName.starts_with("FCSEL") && PrevInst && TII->getName(PrevInst->getOpcode()).starts_with("FCMP")) {
        ++NumInnerLoops_HasFcmpFcsel;
        --LoopCount; // assume the pair would not cross cachelines
        DBG(7, dbgs() << "    Observed FCMP-FCSEL pair\n");
      }

      ++LoopCount;
      if (PrevInst && MI.isBranch() && isCompareBranchFusion(PrevInst, &MI))
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
        DBG_OBSERVED("WFE instruction", OpcodeName);
      }
      // Check for Pointer Authentication (AUT*) instructions
      else if (startsWithAny({"AUT"})) {
        // Pointer Authentication instructions have additional overhead
        ++LoopCount;
        DBG_OBSERVED("Pointer Authentication instruction", OpcodeName);
      }
      // Check for integer MADD (Multiply-Add) instructions
      // MADD, MADDWrrr, MADDXrrr, SMADDL, UMADDL (but not FMADD for floating-point)
      else if (startsWithAny({"MADD", "SMADD", "UMADD"}) && !OpcodeName.starts_with("FMADD")) {
        // Integer multiply-add has additional overhead
        ++LoopCount;
        if (OpcodeName.starts_with("MADD") && (LoopUnrollASMSkip & 0x1))
          DO_SKIP_HELPER(HasMadd);
      }
      // Check for Synchronization Barrier instructions
      // DMB (Data Memory Barrier), DSB (Data Synchronization Barrier), ISB (Instruction Synchronization Barrier)
      else if (startsWithAny({"DMB", "DSB", "ISB"})) {
        // Synchronization barriers have very significant overhead (4 additional cycles)
        LoopCount += 4;
        DBG_OBSERVED("synchronization barrier instruction", OpcodeName);
      }

      if (MI.isIndirectBranch() || (MI.isCall() && MI.getOperand(0).isReg()))
        DO_SKIP_HELPER(HasIndirect);

      // We want to exclude loops with any control flow changing instructions
      // (branches, calls, returns) except for the terminator
      if (MI.isBranch() && !MI.isTerminator())
        InternalBranches.push_back(&MI);

      // Check for call instructions
      if (MI.isCall()) {
        if (LoopUnrollASMSkip & 0x8) {
          DO_SKIP_HELPER(HasCall);
        } else {
          InternalBranches.push_back(&MI);
        }
      }

      // Assert that there are no return instructions (except terminators)
      assert(!MI.isReturn() && "Loop contains return instruction");

      // Check for atomic operations
      if (isAtomicInstruction(MI))
        DO_SKIP_HELPER(HasAtomicOps);

      // Check for vector load/store by-element instructions (LD1, LD2, LD3, LD4, ST1, ST2, ST3, ST4)
      if (startsWithAny({"LD1", "LD2", "LD3", "LD4", "ST1", "ST2", "ST3", "ST4"}) && (LoopUnrollASMSkip & 0x2))
        DO_SKIP_HELPER(HasVectorByElement);

      // Check for RQSRT (Reciprocal Square Root) instructions
      if (OpcodeName.contains_insensitive("rsqrt") && (LoopUnrollASMSkip & 0x4))
        DO_SKIP_HELPER(HasRsqrt);

      // Track previous instruction for fusion detection
      PrevInst = &MI;
    }
  }

  return LoopCount;
}

/// Find sub-loop blocks for Simple_SubLoop pattern detection
SmallVector<MachineBasicBlock *, 4> LoopUnrollASM::findSubLoopBlocks(const SmallVectorImpl<MachineBasicBlock *> &LoopBlocksInOrder,
  MachineInstr *&BackwardBranch, MachineBasicBlock *&NextBlock) {
  assert(!LoopBlocksInOrder.empty() && "LoopBlocksInOrder should not be empty");

  SmallVector<MachineBasicBlock *, 4> TraversedBlocks;

  // Traverse blocks in layout order to find backward conditional branch
  // Find valid sub-loop pattern: sequence ending with backward conditional branch
  for (MachineBasicBlock *MBB : LoopBlocksInOrder) {
    TraversedBlocks.push_back(MBB);
    MachineInstr *LastBranch = nullptr;
    for (MachineInstr &MI : *MBB)
      if (MI.isBranch())
        LastBranch = &MI;

    // Check if the last branch is a backward conditional branch
    if (LastBranch && LastBranch->isConditionalBranch()) {
      // Check if it branches to a previously traversed block
      for (const MachineOperand &MO : LastBranch->operands()) {
        if (MO.isMBB()) {
          MachineBasicBlock *Target = MO.getMBB();
          if (llvm::find(TraversedBlocks, Target) != TraversedBlocks.end()) {
            // Found valid backward conditional branch - collect and return sub-loop blocks immediately
            BackwardBranch = LastBranch;

            // Find the next block after current MBB in LoopBlocksInOrder
            auto CurrentIt = llvm::find(LoopBlocksInOrder, MBB);
            if (CurrentIt != LoopBlocksInOrder.end()) {
              auto NextIt = std::next(CurrentIt);
              NextBlock = (NextIt != LoopBlocksInOrder.end()) ? *NextIt : nullptr;
            } else {
              NextBlock = nullptr;
            }

            SmallVector<MachineBasicBlock *, 4> SubLoopBlocksInOrder;
            bool InRange = false;
            for (MachineBasicBlock *LoopMBB : LoopBlocksInOrder) {
              if (LoopMBB == Target)
                InRange = true;
              if (InRange)
                SubLoopBlocksInOrder.push_back(LoopMBB);
              if (LoopMBB == MBB)
                break;
            }
            return SubLoopBlocksInOrder;
          }
          else {  // LastBranch targets non-traversed block
            if (llvm::find(LoopBlocksInOrder, Target) != LoopBlocksInOrder.end()) {
              // Target is for a block inside the Loop, return with an empty block
              return {};
            }
          }
        }
      }
    }
  }

  // Return empty vector if no sub-loop found
  return {};
}

  // First pass: handles loop-unrolling
bool LoopUnrollASM::processLoop(MachineLoop *Loop, MachineFunction &MF) {
  // Process inner loops first (depth-first)
  bool Changed = false;
  for (MachineLoop *SubLoop : *Loop) {
    Changed |= processLoop(SubLoop, MF);
  }

  DBG(3, {
    SmallVector<MachineBasicBlock *, 4> LoopBlocksInOrder;
    for (MachineBasicBlock &MBB : MF)
      if (Loop->contains(&MBB))
        LoopBlocksInOrder.push_back(&MBB);
    debugPrintLoopInfo(Loop, " Examining", LoopBlocksInOrder);
  });
  // Only process innermost loops
  if (!Loop->getSubLoops().empty()) {
    DBG_SKIP("non-innermost");
    return Changed;
  }
  ++NumInnermostLoops;

  MachineInstr *BackedgeBranch = findLoopBackedgeBranch(Loop);
  unsigned LoopCount = 0;
  SmallVector<MachineInstr *, 4> InternalBranches;
  if (auto OptionalCount = calculateLoopCount(Loop->blocks(), InternalBranches))
    LoopCount = *OptionalCount;
  else
    return Changed; // Helper function indicated we should skip this loop

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
        DBG(4, dbgs() << "  Loop ends with unconditional exit branch\n");
      }
      break; // Found the first valid instruction before backedge
    }
  }

  // Check if this is a loop with multiple terminators where all non-backedge terminators target exit
  DBG(7, {dbgs() << "  info: NumTerminators=" << NumTerminators
                     << " isLoopSimplifyForm=" << isLoopSimplifyForm(Loop)
                     << " endsUncondExitBranch=" << (lastUnconditionalExitBranch != nullptr)
                     << " InternalBranches=" << InternalBranches.size()
                     << "\n";
  });
  DBG(8, {
    for (auto T: Terminators)
      dbgs() << "    info: cond=" << T->isConditionalBranch() <<
                " branchTargetsExit=" << branchTargetsExit(T) <<
                " backedge=" << (T == BackedgeBranch) << " " << *T;
  });

  MachineBasicBlock *Latch = Loop->getLoopLatch();
  if (Pattern == Nonsupported && Latch) {
    MachineBasicBlock::iterator LastIter = Latch->getLastNonDebugInstr();
    if (LastIter == Latch->end())
      DO_SKIP(Invalid);

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
          DBG(4, dbgs() << "  Detected Two_Backedge_Uncond pattern\n");
        }
        if (Last != lastUnconditionalExitBranch) {
          ++NumInnerLoops_InvalidUncondExit;
          DBG(4, dbgs() << "    Last != lastUnconditionalExitBranch\n");
        }
      }
    }

    // Check if the last instruction is actually a branch
    // In some cases (e.g., when there are no terminators), Last may not be a branch
    if (Pattern == Nonsupported && !Last->isBranch()) {
      ++NumInnerLoops_LastInstNotBranch;
      DBG(0, dbgs() << "  warning: Invalid loop: Last instruction is not a branch\n");
      //return Changed;
    }
  }

  // Collect loop blocks in their layout order (not Loop->blocks() order)
  SmallVector<MachineBasicBlock *, 4> LoopBlocksInOrder;
  for (MachineBasicBlock &MBB : MF)
    if (Loop->contains(&MBB))
      LoopBlocksInOrder.push_back(&MBB);

  // Lambda to process a tight loop with given parameters
  MachineBasicBlock *Header = Loop->getHeader();
  auto processTightLoopWithPattern = [&](MachineInstr *BackedgeBranch, TerminatorPattern Pattern,
                                         MachineBasicBlock *ExitBlock, bool EndsWithUncondExit,
                                         const SmallVectorImpl<MachineBasicBlock *> &BlocksInOrder,
                                         unsigned LoopCount) -> bool {
    unsigned NumInsts = countInstructions(make_range(BlocksInOrder.begin(), BlocksInOrder.end()));
    SmallVector<MachineOperand, 4> EmptyCond;
    BackedgeInfo BackedgeInfoObj(BackedgeBranch, Pattern, ExitBlock, EmptyCond, EmptyCond, EndsWithUncondExit);
    return processTightLoop(Loop, Header, LoopCount, BackedgeInfoObj, NumInsts, BlocksInOrder);
  };

  // Lambda to check for Simple_SubLoop pattern
  auto checkSimpleSubLoop = [&]() -> std::optional<bool> {
    if (Pattern == Nonsupported && (LoopUnrollASMEnable & 0x8)) {
      MachineInstr *BackwardBranch = nullptr;
      MachineBasicBlock *NextBlock = nullptr;
      SmallVector<MachineBasicBlock *, 4> SubLoopBlocks = findSubLoopBlocks(LoopBlocksInOrder, BackwardBranch, NextBlock);
      if (!SubLoopBlocks.empty()) {
        DBG(4, dbgs() << "    Matched SubLoop pattern\n");
        SmallVector<MachineInstr *, 4> InternalBranches;
        auto OptionalCount = calculateLoopCount(make_range(SubLoopBlocks.begin(), SubLoopBlocks.end()), InternalBranches);
        assert(OptionalCount && "calculateLoopCount should not fail for SubLoop pattern");
        return processTightLoopWithPattern(BackwardBranch, Simple_SubLoop, NextBlock, false, SubLoopBlocks, *OptionalCount);
      }
    }
    return std::nullopt;
  };

  if (NumTerminators > 1 && Pattern == Nonsupported) {
    // TODO: this may replace the more complex Latch based logic above
    if (Pattern == Nonsupported && lastUnconditionalExitBranch && NumTerminators == 2 && BackedgeBranch) {
      if (LoopUnrollASMEnable & 0x2) {
        Pattern = Two_Backedge_Uncond;
        DBG(4, dbgs() << "  Detected Two_Backedge_Uncond pattern; fixup\n");
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
        DBG(4, dbgs() << "  Detected Multi_CondExit_Backedge pattern ("
                          << NumTerminators << " terminators: exit branches + backedge)\n");
      }
    }

    if (Pattern == Nonsupported) {
      if (auto result = checkSimpleSubLoop(); result) // NumTerminators > 1
        return *result;
      if (NumTerminators == 2)
        ++NumInnerLoopsMultipleTerminators2;
      else if (NumTerminators == 3)
        ++NumInnerLoopsMultipleTerminators3;
      else if (NumTerminators >= 4)
        ++NumInnerLoopsMultipleTerminators4Plus;
      DBG(4, dbgs() << "  skipping: Multiple terminators (" << NumTerminators << ", not accepted pattern)\n");
      return Changed;
    }
  } // NumTerminators > 1

  if (Pattern == Nonsupported)
    if (auto result = checkSimpleSubLoop(); result) // NumTerminators = 1
      return *result;

  // Skip loops with too many basic blocks
  // TODO: count loop with internal branches that create short paths
  if (Loop->getNumBlocks() > LoopUnrollASMMaxBlocks)
    DO_SKIP(TooManyBlocks, "(" << Loop->getNumBlocks() << " > " << LoopUnrollASMMaxBlocks << ")");

  if (!BackedgeBranch)
    DO_SKIP(BranchConditionalNoBackedge);

  SmallVector<MachineBasicBlock *, 4> Latches;
  Loop->getLoopLatches(Latches);
  assert(Latches.size() == 1 && "Loop must have single latch");

  // Classify the branch type and skip if not suitable for unrolling
  // TODO: we should be able to unroll simple cases like:
  // Loop:  inst1
  //        b.eq Exit
  //        b Loop
  //  Exit:
  if (BackedgeBranch->isUnconditionalBranch())
    DO_SKIP(BranchUnconditional);

  assert(BackedgeBranch->isConditionalBranch() && "Loop with non-conditional backedge branch!");

  // Check if this is a single basic block loop (Head == Latch)
  // Exception: Multi_CondExit_Backedge pattern can have multiple blocks
  if (Pattern != Multi_CondExit_Backedge && (!Header || !Latch || Header != Latch))
    DO_SKIP(NotSingleBB);

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
  if (ExitBlocks.empty())
    DO_SKIP(Invalid);
  MachineBasicBlock *ExitBlock = ExitBlocks[0];

  if (!InternalBranches.empty())
    DO_SKIP(HasInternalBranch);

  if (LoopCount >= LoopUnrollASMMaxOps) {
    ++NumInnerLoopsTooManyInsts;
     DBG(4, dbgs() << "  skipping: Too many instructions (" << LoopCount << " >= " << LoopUnrollASMMaxOps << ")\n");
    return Changed;
  }

  // an unconditional exit branch that ends a loop won't be unrolled
  unsigned AdjustedLoopCount = lastUnconditionalExitBranch ? LoopCount - 1 : LoopCount;

  return processTightLoopWithPattern(BackedgeBranch, Pattern, ExitBlock, lastUnconditionalExitBranch != nullptr, LoopBlocksInOrder, AdjustedLoopCount);
}

void LoopUnrollASM::debugPrintLoopInfo(const MachineLoop *Loop,
                                       StringRef Prefix,
                                       const SmallVectorImpl<MachineBasicBlock *> &Blocks,
                                       MachineBasicBlock *ExitBlock) {
  MachineFunction &MF = *Loop->getHeader()->getParent();
  MachineBasicBlock *Header = Loop->getHeader();
  // Get debug location information
  DebugLoc DL;
  for (MachineInstr &MI : *Header) {
    if (!MI.isDebugInstr() && MI.getDebugLoc()) {
      DL = MI.getDebugLoc();
      break;
    }
  }

  DBG(3, {
    dbgs() << Prefix << " loop; ";
    if (DL) {
      dbgs() << "Source location: ";
      if (DL.getLine() != 0 && DL.getScope()) {
        if (auto *Scope = dyn_cast<DIScope>(DL.getScope())) {
          dbgs() << Scope->getFilename() << ":" << DL.getLine();
          if (DL.getCol() != 0)
            dbgs() << ":" << DL.getCol();
        }
      }
    } else {
          dbgs() << Prefix << "in function " << MF.getName();
    }
    dbgs() << "\n";

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
    for (MachineBasicBlock *MBB : Blocks) {
      printBlockName(MBB);
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
// - Have less than N instructions (default 46, configurable via -loop-unroll-asm-max-ops)
// - Are single basic-block loops (Head == Latch)
// - Have a conditional branch as the backedge
// - Loop body has no control flow instructions (branches/calls/returns)
// - Loop body has no atomic operations (ldxr/stxr, atomicrmw, etc.)
bool LoopUnrollASM::processTightLoop(MachineLoop *Loop,
                                     MachineBasicBlock *Header,
                                     unsigned LoopCount,
                                     BackedgeInfo &Backedge,
                                     unsigned LoopInsts,
                                     const SmallVectorImpl<MachineBasicBlock *> &BlocksInOrder) {
  ++NumLoopsDetected;

  const unsigned LoopCycles =
      (LoopCount + MachineWidth - 1) / MachineWidth; // round-up int divide

  unsigned Bubbles = calculateBubbles(LoopCount, MachineWidth);
  // Hanlde the case if the loop would possibly induce +20% Frontend Bound
  if (Bubbles / float(MachineWidth * LoopCycles) > LoopUnrollASMFetchBubblesThreshold) {
    // First, we need to analyze the loop branch to see if we can invert it
    // Analyze the original branch to extract condition
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    if (TII->analyzeBranch(*Backedge.Branch->getParent(), TBB, FBB, Backedge.Cond)) {
      DBG(4, dbgs() << "  Unable to analyze branch for unrolling\n");
      ++NumInnerLoopsBranchPrepFailure;
      return false;
    }

    // Fall-through could vary in case of conditional branch with fall-through (FBB==null)
    if (TBB && !FBB) {
      MachineBasicBlock *FallthroughBB = Backedge.Branch->getParent()->getNextNode();
      if (FallthroughBB && FallthroughBB != Backedge.ExitBlock) {
        DBG(4, dbgs() << "  saw a different fallthrough vs exit blocks\n");
        Backedge.ExitBlock = FallthroughBB;
      }
    }

    // Try to invert the condition
    Backedge.InvertedCond = Backedge.Cond;
    if (TII->reverseBranchCondition(Backedge.InvertedCond)) {
      DBG(4, dbgs() << "  Unable to invert branch condition, skipping unrolling\n");
      ++NumInnerLoopsBranchPrepFailure;
      return false;
    }

    DBG(3, {debugPrintLoopInfo(Loop, " Found qualifying", BlocksInOrder, Backedge.ExitBlock);
      dbgs() << "  with Loop count: " << LoopCount << " for " << LoopInsts << " instructions\n";});
    unsigned UC = findBestUnrollCount(LoopCount, Bubbles);

    LoopInsts = Backedge.Pattern == Two_Backedge_Uncond ? LoopInsts - 1 : LoopInsts;
    duplicateLoopBody(Loop, UC, Backedge, LoopInsts, BlocksInOrder);

    NumAddedInsts += (UC - 1) * LoopInsts;
    ++NumLoopsUnrolled;
    if (Backedge.Pattern == Two_Backedge_Uncond) {
      ++NumLoopsUnrolledCondUncond;
    } else if (Backedge.Pattern == Multi_CondExit_Backedge) {
      ++NumLoopsUnrolledMultiCondExit;
    } else if (Backedge.Pattern == Simple_SubLoop) {
      ++NumLoopsUnrolledSimpleSubLoop;
    }
    return true;
  }
  DBG(4, dbgs() << "  bubbles are good\n");

  ++NumLoopsNotEnoughBubbles;
  return false;
}

unsigned LoopUnrollASM::findBestUnrollCount(unsigned LoopCount,
                                            unsigned Bubbles) {
  unsigned BestUC = 2; // Start with the default unroll factor of 2
  unsigned UC = 2;

  while (true) {
    // Calculate the new loop count after unrolling
    unsigned NewLoopCount = LoopCount * UC;

    // Stop if the new loop count is too large
    if (NewLoopCount > 12 * MachineWidth)
      break;

    // Calculate new bubbles for this unroll count
    unsigned NewBubbles = calculateBubbles(NewLoopCount, MachineWidth);
    // Update best unroll count if we found fewer bubbles
    if (NewBubbles < Bubbles) {
      DBG(10, dbgs() << "     findBestUnrollCount: Bubbles=" << Bubbles << " NewBubbles=" << NewBubbles << " UC=" << UC << "\n");
      BestUC = UC;
      Bubbles = NewBubbles; // Update Bubbles for comparison in next iteration
    }

    // Stop if we've eliminated most bubbles
    if (NewBubbles < LoopUnrollASMMinBubbles && (LoopCount > 4 || UC > 2))
      break;

    ++UC;
  }

  return BestUC;
}

void LoopUnrollASM::duplicateLoopBody(MachineLoop *Loop,
                                      unsigned UnrollFactor,
                                      const BackedgeInfo &Backedge,
                                      unsigned NumInsts,
                                      const SmallVectorImpl<MachineBasicBlock *> &BlocksInOrder) {
  assert (UnrollFactor > 1);

  unsigned OriginalLoopInstCount = (Backedge.Pattern == Simple_SubLoop) ? countInstructions(Loop->blocks()) : 0;
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineBasicBlock *Header = Backedge.Pattern == Simple_SubLoop ? BlocksInOrder[0] : Loop->getHeader();
  MachineFunction *MF = Header->getParent();
  MachineBasicBlock *BackedgeBlock = Backedge.Branch->getParent();

  // Collect all non-terminator instructions to duplicate from all loop blocks
  // Use a map to maintain per-block instruction lists for multi-block loops
  DenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 16>> InstsToClone;
  for (MachineBasicBlock *MBB : BlocksInOrder) {
    SmallVector<MachineInstr *, 16> &BlockInsts = InstsToClone[MBB];
    for (MachineInstr &MI : *MBB) {
      // Skip PHI nodes and debug instructions
      if (MI.isPHI() || MI.isDebugInstr())
        continue;

      // For the backedge block, skip terminators (we'll insert new ones)
      // For other blocks, include terminators (they need to be cloned)
      // For Simple_SubLoop pattern, check conditional branch instead of terminator
      if (MBB == BackedgeBlock &&
          (Backedge.Pattern == Simple_SubLoop ? MI.isConditionalBranch() : MI.isTerminator()))
        continue;

      BlockInsts.push_back(&MI);
    }
  }

  DBG(4, dbgs() << "  Original loop has " << NumInsts << " instructions spanning " << Loop->getNumBlocks() << " blocks.\n");

  // We need to create new basic blocks for proper control flow
  // The structure will be:
  // Header -> Body1 -> Body2 -> ... -> BodyN -> Header (loop back)
  //    |        |        |              |
  //    v        v        v              v
  //  Exit    Exit      Exit           Exit

  // Create UnrollFactor-1 new basic blocks for each original loop block
  // (we reuse the original blocks for the first iteration)
  SmallVector<SmallVector<MachineBasicBlock *, 4>, 4> NewBlocks(UnrollFactor - 1);
  MachineBasicBlock *PrevBlock = BlocksInOrder.back(); // Last block in original loop order

  for (unsigned i = 1; i < UnrollFactor; ++i) {
    // Create a new block for each original loop block in layout order
    for (MachineBasicBlock *OrigMBB : BlocksInOrder) {
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
    for (MachineBasicBlock *OrigMBB : BlocksInOrder) {
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
    } // for BlocksInOrder
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

  // Update successors for original non-backedge blocks who keep their terminators
  // but need CFG successors updated to point to duplicated blocks in the first unrolled iteration
  if (Backedge.Pattern == Multi_CondExit_Backedge) {
    for (unsigned j = 0; j < BlocksInOrder.size(); ++j) {
      MachineBasicBlock *OrigMBB = BlocksInOrder[j];

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
          auto It = llvm::find(BlocksInOrder, OrigSucc);
          if (It != BlocksInOrder.end()) {
            unsigned SuccIdx = std::distance(BlocksInOrder.begin(), It);
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
      MachineBasicBlock *OrigMBB = BlocksInOrder[j];
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
            auto It = llvm::find(BlocksInOrder, OrigSucc);
            if (It != BlocksInOrder.end()) {
              unsigned SuccIdx = std::distance(BlocksInOrder.begin(), It);
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

  DBG(4, dbgs() << "  Duplicated loop body with unroll factor " << UnrollFactor << "\n");

  // Verify instruction count matches expectations using helper function
  unsigned NewLoopInstCount = countInstructions(Loop->blocks());
  unsigned ExpectedInstCount = (Backedge.Pattern == Simple_SubLoop)
                                 ? OriginalLoopInstCount + (UnrollFactor - 1) * NumInsts
                                 : UnrollFactor * NumInsts;
  if (NewLoopInstCount != ExpectedInstCount && NewLoopInstCount != (ExpectedInstCount+1)) {
    dbgs() << "ERROR: Instruction count mismatch after loop unrolling!\n";
    dbgs() << "  Pattern: " << Backedge.Pattern << " EndsWithUncondExit=" << Backedge.EndsWithUncondExit << "\n";
    dbgs() << "  Expected: " << ExpectedInstCount
           << " (UnrollFactor=" << UnrollFactor << " * NumInsts=" << NumInsts << ")\n";
    dbgs() << "  Actual: " << NewLoopInstCount << "\n";

    debugPrintLoopInfo(Loop, " ", BlocksInOrder, Backedge.ExitBlock);

    dbgs() << "  Instruction counts per block of original loop:\n";
    for (MachineBasicBlock *MBB : BlocksInOrder) {
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

unsigned LoopUnrollASM::countInstructions(const iterator_range<MachineLoop::block_iterator> Blocks) {
  // Count # instructions in blocks
  unsigned LoopInsts = 0;
  for (MachineBasicBlock *MBB : Blocks)
    for (MachineInstr &MI : *MBB)
      if (!MI.isDebugInstr() && !MI.isPseudo())
        ++LoopInsts;
  return LoopInsts;
}

bool LoopUnrollASM::areLoopBlocksContiguous(const MachineLoop *Loop) {
  if (Loop->blocks().empty())
    return true;

  MachineFunction *MF = Loop->getHeader()->getParent();

  // Find the first and last loop blocks in layout order
  MachineBasicBlock *FirstLoopBlock = nullptr;
  MachineBasicBlock *LastLoopBlock = nullptr;

  for (MachineBasicBlock &MBB : *MF) {
    if (Loop->contains(&MBB)) {
      if (!FirstLoopBlock)
        FirstLoopBlock = &MBB;
      LastLoopBlock = &MBB;
    }
  }

  if (!FirstLoopBlock || !LastLoopBlock)
    return false;

  // Count blocks between first and last (inclusive)
  unsigned BlocksInRange = 0;
  bool InRange = false;

  for (MachineBasicBlock &MBB : *MF) {
    if (&MBB == FirstLoopBlock)
      InRange = true;

    if (InRange) {
      ++BlocksInRange;

      if (&MBB == LastLoopBlock)
        break;
    }
  }

  // Contiguous if the number of blocks in range equals the number of loop blocks
  return BlocksInRange == Loop->getNumBlocks();
}

  // Second pass: handles alignment
bool LoopUnrollASM::analyzeMachineInsts(MachineFunction &MF) {
  // Get function alignment information from MachineFunction (actual assembly-level alignment)
  const Function &F = MF.getFunction();
  Align FuncAlign = MF.getAlignment();
  unsigned AlignmentValue = FuncAlign.value();

  // Also check the IR Function alignment; Use the larger
  MaybeAlign IRAlign = F.getAlign();
  unsigned IRAlignmentValue = IRAlign ? IRAlign->value() : 1;
  AlignmentValue = std::max(AlignmentValue, IRAlignmentValue);

  DBG(1, dbgs() << " Examining alignment. Function alignment: " << AlignmentValue << " bytes"
                << ", IR alignment: " << IRAlignmentValue << "\n");

  // Track function alignment statistics
  if (AlignmentValue != 1 && AlignmentValue != 4) {
    switch (AlignmentValue) {
      case 8: ++NumFunctionsAlign8; break;
      case 16: ++NumFunctionsAlign16; break;
      case 32: ++NumFunctionsAlign32; break;
      case 64: ++NumFunctionsAlign64; break;
      default: ++NumFunctionsAlignOther; break;
    }
  }

  for (MachineLoop *Loop : *MLI) {
    SmallVector<MachineLoop*, 4> InnerLoops;
    std::function<void(MachineLoop*)> collectInnerLoops = [&](MachineLoop *L) {
      if (L->getSubLoops().empty()) InnerLoops.push_back(L);
      else for (MachineLoop *SubLoop : L->getSubLoops()) collectInnerLoops(SubLoop);
    };
    collectInnerLoops(Loop);

    for (MachineLoop *InnerLoop : InnerLoops) {
      MachineBasicBlock *Header = InnerLoop->getHeader();
      if (!Header) continue;
      unsigned LoopAlignment = Header->getAlignment().value();
      const unsigned LoopInsts = countInstructions(InnerLoop->blocks());
      if (LoopAlignment == 1 && LoopUnrollASMAlignThreshold >= 0 &&
          calculateBubbles(LoopInsts, 16) <= (unsigned)LoopUnrollASMAlignThreshold &&
          areLoopBlocksContiguous(InnerLoop)) {
        Header->setAlignment(Align(64));
        LoopAlignment = 64;
        ++NumInnermostLoopsAligned;
        DBG(4, dbgs() << "  Set 64-byte alignment for BB" << Header->getNumber()
          << " who left " << 4*calculateBubbles(LoopInsts, 16) << " untilized bytes\n");
      }

      if (LoopAlignment == 1) continue;
      switch (LoopAlignment) {
        case 8: ++NumInnermostLoopsAlign8; break;
        case 16: ++NumInnermostLoopsAlign16; break;
        case 32: ++NumInnermostLoopsAlign32; break;
        case 64: ++NumInnermostLoopsAlign64; break;
        default: ++NumInnermostLoopsAlignOther; break;
      }
      DBG(3, dbgs() << "  Innermost loop header BB" << Header->getNumber()
                    << " has " << LoopAlignment << "-byte alignment\n");
    }
  }

  // Early exit if requested
  if (!LoopUnrollASMAlignByDirective)
    return false;

  // Early exit if function alignment <= 4
  if (AlignmentValue <= 4)
    return false;

  // Assert that we have a valid alignment
  assert(AlignmentValue > 4 && AlignmentValue <= 64 &&
         "Function alignment must be between 8 and 64 bytes (powers of 2)");
  assert(isPowerOf2_32(AlignmentValue) &&
         "Function alignment must be a power of 2");

  // Without the actual address, we can only know possible offsets from 64-byte boundary
  // For example, if function is 16-byte aligned, it could start at offsets: 0, 16, 32, 48
  // relative to a 64-byte boundary

  // For analysis purposes, we'll check all possible offsets
  // But for FCMP/FCSEL analysis, what matters is relative offset within function

  // Restructured: FuncStartOffset as outer loop to properly track alignment changes across MBBs
  // This ensures that alignment changes for one FuncStartOffset are accounted for in subsequent calculations
  bool MadeChanges = false; // Track if any alignment changes were made

  for (unsigned FuncStartOffset = 0; FuncStartOffset < 64; FuncStartOffset += AlignmentValue) {
    // Reset state for each FuncStartOffset iteration
    unsigned CurrentIndex = 0;
    MachineInstr* PrevInstr = nullptr;

    for (MachineBasicBlock &MBB : MF) {
      // Skip blocks that are not in innermost loops if LoopUnrollASMAlignAll is off
      if (!LoopUnrollASMAlignAll) {
        assert(0 && "LoopUnrollASMAlignAll=0 misses Offset adjustment!");
        MachineLoop *Loop = MLI->getLoopFor(&MBB);
        if (!Loop || !Loop->getSubLoops().empty()) {
          // Skip if not in a loop or not in an innermost loop
          DBG(5, dbgs() << "Skipping MBB " << MBB.getName()
                           << " - not in innermost loop\n");
          ++NumAlignMBBsSkippedNotInnermost;
          continue;
        }
      }

      // Reset previous instruction state at start of each MBB to prevent cross-MBB detection
      PrevInstr = nullptr;

      // Handle MBB alignment by adjusting CurrentIndex for alignment padding
      if (unsigned MBBAlign = MBB.getAlignment().value(); MBBAlign > 4) {
        // Calculate potential padding due to MBB alignment
        // CurrentIndex is in 4-byte instruction units, convert to bytes for alignment calc
        unsigned CurrentByteOffset = CurrentIndex * 4 + FuncStartOffset;
        unsigned AlignedOffset = alignTo(CurrentByteOffset, MBBAlign);
        unsigned PaddingBytes = AlignedOffset - CurrentByteOffset;
        // Convert padding back to instruction units (divide by 4)
        CurrentIndex += (PaddingBytes / 4);
        DBG(6, dbgs() << "  MBB " << MBB.getNumber() << " has " << MBBAlign
                      << "-byte alignment, added " << (PaddingBytes / 4)
                      << " instruction slots for padding\n");
      }

      for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
        MachineInstr &MI = *I;

        // Skip debug instructions and pseudo instructions
        if (MI.isDebugInstr() || MI.isPseudo())
          continue;

        DBG(8, dbgs() << "     InstOffset=" << (CurrentIndex + FuncStartOffset / 4)
                      << " BB=" << MBB.getNumber()
                      << " FuncOffset=" << FuncStartOffset
                      << " " << MI);

        StringRef OpcodeName = TII->getName(MI.getOpcode());

        // Check if previous instruction was FCMP and current is FCSEL
        if (PrevInstr && (CurrentIndex % 2) == 0 && OpcodeName.starts_with("FCSEL")
            && TII->getName(PrevInstr->getOpcode()).starts_with("FCMP")) {
          // Calculate actual offsets including function start offset
          unsigned FCMPActualOffset = (CurrentIndex - 1) * 4 + FuncStartOffset;
          unsigned FCSELActualOffset = CurrentIndex * 4 + FuncStartOffset;

          // Check if FCMP and FCSEL cross cacheline boundaries (cacheline is offset / 16)
          if ((FCMPActualOffset / 16) != (FCSELActualOffset / 16)) {
            DBG(6, {
              dbgs() << "    FuncStartOffset=" << FuncStartOffset << " CurrentIndex=" << CurrentIndex << " FoundCrossing=1 BB=" << MBB.getNumber() << "\n";
              dbgs() << "    " << MI;
              // Find the next real (non-debug, non-pseudo) instruction
              auto NextI = std::next(I);
              while (NextI != E && (NextI->isDebugInstr() || NextI->isPseudo()))
                ++NextI;
              if (NextI != E)
                dbgs() << "    " << *NextI;
            });

            // Find target for alignment: innermost loop header or MBB itself
            MachineBasicBlock *AlignTarget = &MBB;

            MachineLoop *CurrentLoop = MLI->getLoopFor(&MBB);
            if (CurrentLoop && CurrentLoop->getSubLoops().empty()) {
              // MBB is in an innermost loop - align first BB in layout order for better I-cache alignment

              MachineBasicBlock *LoopHeader = CurrentLoop->getHeader();
              MachineBasicBlock *FirstBBInLayout = nullptr;

              // Find first loop block in layout order
              for (MachineBasicBlock &LayoutMBB : MF) {
                if (CurrentLoop->contains(&LayoutMBB)) {
                  FirstBBInLayout = &LayoutMBB;
                  break;
                }
              }

              // FirstBBInLayout must always be found since CurrentLoop is non-empty
              assert(FirstBBInLayout && "FirstBBInLayout must be found for non-empty loop");
              AlignTarget = FirstBBInLayout;
              DBG(6, dbgs() << "  Setting alignment on first loop BB in layout order BB"
                            << AlignTarget->getNumber() << " (header: BB" << LoopHeader->getNumber() << ")\n");
            } else {
              DBG(6, dbgs() << "  Setting alignment on MBB " << AlignTarget->getNumber() << "\n");
            }

            // Calculate minimum alignment to prevent FCMP-FCSEL straddling
            // FCMP+FCSEL need 8 bytes total, cacheline is 16 bytes
            // Use 16-byte alignment to ensure proper placement
            unsigned RequiredAlign = 16;
            unsigned CurrentAlign = AlignTarget->getAlignment().value();

            if (CurrentAlign < RequiredAlign) {
              AlignTarget->setAlignment(Align(RequiredAlign));
              DBG(6, dbgs() << "  Set " << RequiredAlign << "-byte alignment on BB"
                            << AlignTarget->getNumber() << " to prevent FCMP-FCSEL straddling\n");
              ++NumAlignmentsSetForStraddle;
              MadeChanges = true;
            } else {
              DBG(6, dbgs() << "  BB" << AlignTarget->getNumber() << " already has sufficient alignment ("
                            << CurrentAlign << " >= " << RequiredAlign << ")\n");
            }

            // Continue to check for more crossings in this function
          }
        }

        // Update previous instruction info for next iteration
        PrevInstr = &MI;

        CurrentIndex += 1; // Increment by 1 instruction (multiply by 4 for byte offset when needed)
      } // for MachineInstr
    } // for MachineBasicBlock

    // If we found any crossing for this FuncStartOffset, we're done
    // No need to check other FuncStartOffsets
    if (MadeChanges) {
      break;
    }
  } // for FuncStartOffset

  return MadeChanges;
}
