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

STATISTIC(NumLoopsDetected, "Number of tight loops detected by LoopUnrollASM");
STATISTIC(NumLoopsUnrolled, "Number of tight loops unrolled by LoopUnrollASM");
STATISTIC(NumLoopsUnrolledCondUncond, "Number of tight loops unrolled with cond+uncond terminators");
STATISTIC(NumLoopsUnrolledMultiCondExit, "Number of tight loops unrolled with multi-cond-exit+backedge pattern");
STATISTIC(NumLoopsFailedCondInversion, "Number of loops skipped due to branch inversion failure");
STATISTIC(NumLoopsNotEnoughBubbles, "Number of tight loops skipped (not enough bubbles)");
STATISTIC(NumInnerLoopsNotSingleLatch, "Number of inner loops skipped (not single latch)");
STATISTIC(NumInnerLoopsMultipleTerminators, "Number of inner loops skipped (multiple terminators)");
STATISTIC(NumInnerLoopsMultipleTerminators2, "Number of inner loops skipped (2 terminators, non-standard)");
STATISTIC(NumInnerLoopsMultipleTerminators3, "Number of inner loops skipped (3 terminators)");
STATISTIC(NumInnerLoopsMultipleTerminators4Plus, "Number of inner loops skipped (4+ terminators)");
STATISTIC(NumInnermostLoops, "Number of inner loops skipped (not innermost)");
STATISTIC(NumInnerLoopsInvalidTerminator, "Number of inner loops skipped (invalid terminator instruction)");
STATISTIC(NumInnerLoopsBranchUnconditional, "Number of inner loops skipped (unconditional branch)");
STATISTIC(NumInnerLoopsBranchUnconditionalWithSingleCondInternal, "Number of inner loops skipped (unconditional branch with single conditional internal)");
STATISTIC(NumInnerLoopsBranchIndirect, "Number of inner loops skipped (indirect branch)");
STATISTIC(NumInnerLoopsBranchConditionalNoBackedge, "Number of inner loops skipped (conditional branch without backedge)");
STATISTIC(NumInnerLoopsNotSingleBB, "Number of inner loops skipped (Header != Latch)");
STATISTIC(NumInnerLoopsHasAtomicOps, "Number of inner loops skipped (has atomic ops)");
STATISTIC(NumInnerLoopsHasInternalBranch, "Number of inner loops skipped (has internal branch)");
STATISTIC(NumInnerLoopsTooManyInsts, "Number of inner loops skipped (too many instructions)");
STATISTIC(NumInnerLoopsCannotAnalyzeBranch, "Number of tight loops skipped (cannot analyze branch)");

static cl::opt<unsigned> LoopUnrollASMMaxInsts(
    "loop-unroll-asm-max-insts",
    cl::desc("Maximum number of instructions in a loop for LoopUnrollASM to process"),
    cl::init(46), cl::Hidden);

namespace {
enum TerminatorPattern {
  Nonsupported = 0,
  One_Backedge,
  Two_Backedge_Uncond,
  Multi_CondExit_Backedge
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
  // Static helper function to calculate bubbles
  static unsigned calculateBubbles(unsigned LoopCount) {
    const unsigned MachineWidth = 10;
    unsigned Remainder = LoopCount % MachineWidth;
    return Remainder ? MachineWidth - Remainder : 0;
  }

  void debugPrintLoopInfo(MachineFunction &MF, MachineBasicBlock *Header,
                          StringRef Prefix);
  MachineInstr* findLoopBackedgeBranch(const MachineLoop* Loop);
  bool processLoop(MachineLoop *Loop, MachineFunction &MF);
  bool processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                        MachineBasicBlock *Header, unsigned LoopCount,
                        TerminatorPattern Pattern, MachineInstr *BackedgeBranch);
  unsigned findBestUnrollCount(unsigned LoopCount, unsigned Bubbles,
                               unsigned MachineWidth);
  void duplicateLoopBody(MachineLoop *Loop, unsigned UnrollFactor,
                         const SmallVectorImpl<MachineOperand> &InvertedCond,
                         MachineInstr *BackedgeBranch, TerminatorPattern Pattern);
};
} // end anonymous namespace

char LoopUnrollASM::ID = 0;
char &llvm::LoopUnrollASMID = LoopUnrollASM::ID;

INITIALIZE_PASS_BEGIN(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopUnrollASM, DEBUG_TYPE, "Loop Unroll at Assembly Level", false, false)

bool LoopUnrollASM::runOnMachineFunction(MachineFunction &MF) {
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

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
  assert(Loop);
  MachineBasicBlock* Header = Loop->getHeader();
  assert(Header);

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
  if (!MainBackedgeBlock) {
    return nullptr;
  }

  // Find the terminating branch instruction in the backedge block
  MachineInstr* BackedgeBranch = nullptr;
  // Look for the last branching instruction in the block
  for (auto I = MainBackedgeBlock->rbegin(); I != MainBackedgeBlock->rend(); ++I) {
    MachineInstr& MI = *I;

    if (!MI.isBranch())
      continue;

    // Verify this branch actually targets the loop header
    bool BranchesToHeader = false;

    if (MI.isUnconditionalBranch()) {
      // For unconditional branches, check if it targets the header
      for (const MachineOperand& MO : MI.operands()) {
        if (MO.isMBB() && MO.getMBB() == Header) {
          BranchesToHeader = true;
          break;
        }
      }
    } else if (MI.isConditionalBranch()) {
      // For conditional branches, check if one of the targets is the header
      for (const MachineOperand& MO : MI.operands()) {
        if (MO.isMBB() && MO.getMBB() == Header) {
          BranchesToHeader = true;
          break;
        }
      }

      // Also check fall-through case
      if (!BranchesToHeader) {
        MachineBasicBlock* FallThrough = MainBackedgeBlock->getNextNode();
        if (FallThrough == Header) {
          BranchesToHeader = true;
        }
      }
    }

    if (BranchesToHeader) {
      BackedgeBranch = &MI;
      break;
    }
  }

  if (!BackedgeBranch) {
    LLVM_DEBUG(dbgs() << "Warning: No branch instruction found that targets loop header\n");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "  info: main backedge branch: " << *BackedgeBranch);

  return BackedgeBranch;
}

bool LoopUnrollASM::processLoop(MachineLoop *Loop, MachineFunction &MF) {
  // Process inner loops first (depth-first)
  bool Changed = false;
  for (MachineLoop *SubLoop : *Loop) {
    Changed |= processLoop(SubLoop, MF);
  }

  MachineBasicBlock *Header = Loop->getHeader();
  LLVM_DEBUG(debugPrintLoopInfo(MF, Header, "Examining"));

  // Only process innermost loops
  if (!Loop->getSubLoops().empty()) {
    LLVM_DEBUG(dbgs() << "  skipping: Not innermost\n");
    return Changed;
  }
  ++NumInnermostLoops;

  MachineBasicBlock *Latch = Loop->getLoopLatch();
  SmallVector<MachineBasicBlock *, 4> Latches;
  Loop->getLoopLatches(Latches);
  if (Latches.size() != 1) {
    ++NumInnerLoopsNotSingleLatch;
    LLVM_DEBUG(dbgs() << "  skipping: Not single latch\n");
    return Changed;
  }
  assert(Latch == Latches[0]);

  // Count instructions in the loop and check for internal branches
  // We want to skip loops that have branches within the loop body
  // (excluding the terminator back-edge branch)
  MachineInstr *BackedgeBranch = findLoopBackedgeBranch(Loop);
  unsigned LoopCount = 0;
  SmallVector<MachineInstr *, 4> InternalBranches;
#ifndef LOOPUNROLLASM_ALLOW_ATOMIC_UNROLL
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool hasAtomicOps = false;
#endif

  // Calculate a precise LoopCount based on the loop size (in # fetch slots)
  // TODO: this should eventually interface with TTI
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      // Skip debug instructions and pseudo instructions
      if (MI.isDebugInstr() || MI.isPseudo())
        continue;

      ++LoopCount;

      // Check if this is a post-index memory operation
      // Post-index operations update the base register and are typically
      // more complex, so we count them with a weight of 2
      if (MI.mayLoadOrStore() && !MI.isTerminator()) {
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
          // Count post-index operations twice (weight = 2)
          ++LoopCount;
        }
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
    }
  }

  MachineBasicBlock::iterator LastIter = Latch->getLastNonDebugInstr();
  if (LastIter == Latch->end()) {
    ++NumInnerLoopsInvalidTerminator;
    LLVM_DEBUG(dbgs() << "  skipping: No non-debug instructions\n");
    return Changed; // No non-debug instructions
  }
  MachineInstr *Last = &*LastIter;

  // Lambda to check if a branch instruction targets an exit block
  auto branchTargetsExit = [&](const MachineInstr *MI) -> bool {
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isMBB()) {
        MachineBasicBlock *Target = MO.getMBB();
        if (Target != Header && !Loop->contains(Target))
          return true;
      }
    }
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
  TerminatorPattern Pattern = NumTerminators == 1 ? One_Backedge : Nonsupported;

  // Check if this is a loop with multiple terminators where all non-backedge terminators target exit
  // Pattern: Multiple conditional exit branches + One backedge = Multi_CondExit_Backedge
  LLVM_DEBUG({dbgs() << "  info: NumTerminators=" << NumTerminators
                     << " Backedge=";
    if (BackedgeBranch)
      BackedgeBranch->print(dbgs());
    else
      dbgs() << "none\n";
    for (auto T: Terminators)
      dbgs() << "    info: Cond=" << T->isConditionalBranch() << " branchTargetsExit=" << branchTargetsExit(T) << "\n";
  });

  // Check if all terminators except BackedgeBranch target the exit
  if (NumTerminators >= 2 && BackedgeBranch) {
    bool allOthersTargetExit = true;
    for (auto T : Terminators) {
      if (T != BackedgeBranch && !branchTargetsExit(T)) {
        allOthersTargetExit = false;
        break;
      }
    }
    if (allOthersTargetExit) {
      Pattern = Multi_CondExit_Backedge;
      LLVM_DEBUG(dbgs() << "  Detected Multi_CondExit_Backedge pattern ("
                        << NumTerminators << " terminators: exit branches + backedge)\n");
    }
  }

  // Check for backedge conditional followed by unconditional exit in the Latch block
  // TODO: tweak the loop_with_cond_uncond_pattern to be detected here (avoid Branch Folding optimization)
  MachineBasicBlock::iterator FirstTermIter = Latch->getFirstTerminator();
  if (FirstTermIter != LastIter) {
    auto SecondTermIter = std::next(FirstTermIter);
    if (SecondTermIter == LastIter &&
        FirstTermIter != Latch->end() &&
        FirstTermIter->isConditionalBranch() &&
        Last->isUnconditionalBranch()) {
      // Pattern 1: conditional branch followed by unconditional branch
      Pattern = Two_Backedge_Uncond;
      LLVM_DEBUG(dbgs() << "  Detected two-terminator pattern (cond + uncond)\n");
    }
  }

  if (Pattern == Nonsupported) {
    ++NumInnerLoopsMultipleTerminators;
    // Increment the specific counter based on number of terminators
    if (NumTerminators == 2)
      ++NumInnerLoopsMultipleTerminators2;
    else if (NumTerminators == 3)
      ++NumInnerLoopsMultipleTerminators3;
    else if (NumTerminators >= 4)
      ++NumInnerLoopsMultipleTerminators4Plus;
    LLVM_DEBUG(dbgs() << "  skipping: Multiple terminators (" << NumTerminators << ", not accepted pattern)\n");
    return Changed;
  }

  if (!BackedgeBranch) {
    ++NumInnerLoopsBranchConditionalNoBackedge;
    LLVM_DEBUG(dbgs() << "  skipping: Loop has no backedge branch\n");
    return Changed;
  }

  // Check if the last instruction is actually a branch
  // In some cases (e.g., when there are no terminators), Last may not be a branch
  if (!Last->isBranch()) {
    ++NumInnerLoopsInvalidTerminator;
    LLVM_DEBUG(dbgs() << "  skipping: Last instruction is not a branch\n");
    return Changed;
  }

  // Classify the branch type and skip if not suitable for unrolling
  if (BackedgeBranch->isUnconditionalBranch()) {
    ++NumInnerLoopsBranchUnconditional;
    // Sub-case: check if there's a single internal conditional branch
    if (InternalBranches.size() == 1 && InternalBranches[0]->isConditionalBranch())
      ++NumInnerLoopsBranchUnconditionalWithSingleCondInternal;
    LLVM_DEBUG(dbgs() << "  skipping: Unconditional branch\n");
    return Changed;
  }

  if (Last->isIndirectBranch()) {
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

  // For Two_Backedge_Uncond pattern, decrement LoopCount since the unconditional
  // branch is not part of the actual loop body
  unsigned AdjustedLoopCount = (Pattern == Two_Backedge_Uncond) ? LoopCount - 1 : LoopCount;

  // Process the tight loop
  return processTightLoop(Loop, MF, Header, AdjustedLoopCount, Pattern, BackedgeBranch);
}

void LoopUnrollASM::debugPrintLoopInfo(MachineFunction &MF,
                                       MachineBasicBlock *Header,
                                       StringRef Prefix) {
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
      if (DL.getLine() != 0) {
        auto *Scope = cast<DIScope>(DL.getScope());
        dbgs() << Scope->getFilename() << ":" << DL.getLine();
        if (DL.getCol() != 0)
          dbgs() << ":" << DL.getCol();
      }
      dbgs() << "\n";
    }
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
                                     unsigned LoopCount, TerminatorPattern Pattern,
                                     MachineInstr *BackedgeBranch) {
  LLVM_DEBUG({debugPrintLoopInfo(MF, Header, "Found qualifying");
    dbgs() << "  with Loop count: " << LoopCount << "\n";});

  ++NumLoopsDetected;

  // TODO: get MachineWidth from TTI / SchedModel
  const unsigned MachineWidth = 10;
  const unsigned LoopCycles =
      (LoopCount + MachineWidth - 1) / MachineWidth; // round-up int divide

  unsigned Bubbles = calculateBubbles(LoopCount);
  // Hanlde the case if the loop would possibly induce +20% Frontend Bound
  if (Bubbles / float(MachineWidth * LoopCycles) > 0.2f) {
    // First, we need to analyze the loop branch to see if we can invert it
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

    // Analyze the original branch to extract condition
    SmallVector<MachineOperand, 4> Cond;
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;

    if (TII->analyzeBranch(*BackedgeBranch->getParent(), TBB, FBB, Cond)) {
      LLVM_DEBUG(dbgs() << "  Could not analyze branch for unrolling\n");
      ++NumInnerLoopsCannotAnalyzeBranch;
      return false;
    }

    // Try to invert the condition
    SmallVector<MachineOperand, 4> InvertedCond(Cond);
    if (TII->reverseBranchCondition(InvertedCond)) {
      LLVM_DEBUG(dbgs() << "  Unable to invert branch condition, skipping unrolling\n");
      ++NumLoopsFailedCondInversion;
      return false;
    }

    // Determine the best unroll factor based on minimizing bubbles
    unsigned UnrollFactor =
        findBestUnrollCount(LoopCount, Bubbles, MachineWidth);

    // Duplicate the loop body with the calculated unroll factor
    duplicateLoopBody(Loop, UnrollFactor, InvertedCond, BackedgeBranch, Pattern);
    ++NumLoopsUnrolled;
    if (Pattern == Two_Backedge_Uncond) {
      ++NumLoopsUnrolledCondUncond;
    } else if (Pattern == Multi_CondExit_Backedge) {
      ++NumLoopsUnrolledMultiCondExit;
    }
    return true;
  }

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
                                      const SmallVectorImpl<MachineOperand> &InvertedCond,
                                      MachineInstr *BackedgeBranch,
                                      TerminatorPattern Pattern) {
  assert (UnrollFactor > 1);

  MachineBasicBlock *Header = Loop->getHeader();
  MachineFunction *MF = Header->getParent();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  MachineBasicBlock *BackedgeBlock = BackedgeBranch->getParent();

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
  assert(!ExitBlocks.empty() && ExitBlocks.size() >= 1 && "Could not find loop exit block");
  MachineBasicBlock *ExitBlock = ExitBlocks[0];

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

  // Analyze the branch in the backedge block to extract condition
  SmallVector<MachineOperand, 4> Cond;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  const bool failed = TII->analyzeBranch(*BackedgeBlock, TBB, FBB, Cond);
  assert(!failed && "should be able to analyze branch here");

  LLVM_DEBUG({
    unsigned TotalInsts = 0;
    for (const auto &Entry : InstsToClone)
      TotalInsts += Entry.second.size();
    dbgs() << "  Original loop body has " << TotalInsts
                    << "  instructions (without Backedge) spanning "
                    << Loop->getNumBlocks() << " block(s).\n";
  });

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
      // These registers are needed by the cloned instructions
      for (const auto &LI : OrigMBB->liveins()) {
        NewBB->addLiveIn(LI.PhysReg);
      }

      NewBlocks[i - 1].push_back(NewBB);
      PrevBlock = NewBB;
    }
  }

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
          MachineBasicBlock *NextBlock = (i == 0) ? NewBlocks[0][0] : NewBlocks[i][0];

          // Check if the next block is the layout successor (fallthrough)
          if (CurrentBlock->isLayoutSuccessor(NextBlock)) {
            // Only insert conditional branch to exit, fall through to next block
            TII->insertBranch(*CurrentBlock, ExitBlock, nullptr, InvertedCond,
                              BackedgeBranch->getDebugLoc());
          } else {
            // Need both branches
            TII->insertBranch(*CurrentBlock, ExitBlock, NextBlock, InvertedCond,
                              BackedgeBranch->getDebugLoc());
          }
        } else {
          // For the last iteration, branch back to header with inverted condition
          // (loop back when we should continue, exit otherwise)
          if (CurrentBlock->isLayoutSuccessor(ExitBlock)) {
            // Only insert conditional branch to header, fall through to exit
            TII->insertBranch(*CurrentBlock, Header, nullptr, Cond,
                              BackedgeBranch->getDebugLoc());
          } else {
            // Need both branches
            TII->insertBranch(*CurrentBlock, Header, ExitBlock, Cond,
                              BackedgeBranch->getDebugLoc());
          }
        }
      }

      BlockIdx++;
    } // for LoopBlocksInOrder
  }

  // Update the CFG - fix successor relationships
  // First update the backedge block's successors (last block of first iteration)
  if (!NewBlocks.empty()) {
    // Backedge block now branches to either first block of second iteration or exit
    // Remove self-loop to header if it exists (only for single-block loops where backedge is in header)
    if (BackedgeBlock->isSuccessor(Header))
      BackedgeBlock->removeSuccessor(Header);
    if (!BackedgeBlock->isSuccessor(NewBlocks[0][0]))
      BackedgeBlock->addSuccessor(NewBlocks[0][0]);
    // ExitBlock successor should already be there
  }

  // Now add successors for the new iteration blocks
  // Each iteration has multiple blocks in the same order as original loop
  for (unsigned i = 0; i < NewBlocks.size(); ++i) {
    for (unsigned j = 0; j < NewBlocks[i].size(); ++j) {
      MachineBasicBlock *BB = NewBlocks[i][j];
      bool isLastBlockInIteration = (j == NewBlocks[i].size() - 1);

      if (isLastBlockInIteration) {
        // Last block in iteration can exit
        if (!BB->isSuccessor(ExitBlock))
          BB->addSuccessor(ExitBlock);

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
        // Non-last blocks within an iteration may need internal successors
        // but these are typically handled by fall-through or existing branches
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Duplicated loop body with unroll factor " << UnrollFactor << "\n");
}

FunctionPass *llvm::createLoopUnrollASMPass() {
  return new LoopUnrollASM();
}
