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

STATISTIC(NumLoopsDetectedASM,
          "Number of tight loops detected by LoopUnrollASM");
STATISTIC(NumLoopsUnrolledASM,
          "Number of tight loops unrolled by LoopUnrollASM");

static cl::opt<unsigned> LoopUnrollASMMaxInsts(
    "loop-unroll-asm-max-insts",
    cl::desc("Maximum number of instructions in a loop for LoopUnrollASM to process"),
    cl::init(46), cl::Hidden);

namespace {
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
  bool processLoop(MachineLoop *Loop, MachineFunction &MF);
  bool processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                        MachineBasicBlock *Header, unsigned InstrCount);
  void duplicateLoopBody(MachineBasicBlock *Header, unsigned UnrollFactor);
  unsigned countLoopInstructions(MachineLoop *Loop);
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

bool LoopUnrollASM::processLoop(MachineLoop *Loop, MachineFunction &MF) {
  // Process inner loops first (depth-first)
  bool Changed = false;
  for (MachineLoop *SubLoop : *Loop) {
    Changed |= processLoop(SubLoop, MF);
  }

  // Only process innermost loops
  if (!Loop->getSubLoops().empty()) {
    return Changed;
  }

  // Check if this is a single basic block loop (Head == Latch)
  MachineBasicBlock *Header = Loop->getHeader();
  MachineBasicBlock *Latch = Loop->getLoopLatch();
  if (!Header || !Latch || Header != Latch)
    return Changed;

  // Check if the loop has a conditional branch back edge
  // Note: The conditional branch might not be the last instruction if there's
  // also an unconditional branch to the exit block
  bool hasConditionalBranch = false;
  for (auto I = Header->rbegin(); I != Header->rend(); ++I) {
    if (I->isDebugInstr())
      continue;
    if (I->isConditionalBranch()) {
      // Check if this conditional branch targets the loop header
      for (MachineOperand &MO : I->operands()) {
        if (MO.isMBB() && MO.getMBB() == Header) {
          hasConditionalBranch = true;
          break;
        }
      }
      if (hasConditionalBranch)
        break;
    }
    // Stop looking after we've seen all terminators
    if (!I->isTerminator())
      break;
  }

  if (!hasConditionalBranch) {
    LLVM_DEBUG(
        dbgs() << "  Loop does not have a conditional branch back to header\n");
    return Changed;
  }

  // Count instructions in the loop
  unsigned InstrCount = countLoopInstructions(Loop);
  if (InstrCount >= LoopUnrollASMMaxInsts)
    return Changed;

  // Process the tight loop
  return processTightLoop(Loop, MF, Header, InstrCount);
}

// processTightLoop() targets certain loops that meet these conditions:
// - Are Inner-most loops
// - Have less than N instructions (default 50, configurable via -loop-unroll-asm-max-insts)
// - Are single basic-block loops (Head == Latch)
// - Have a conditional branch as the backedge
bool LoopUnrollASM::processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                                     MachineBasicBlock *Header,
                                     unsigned InstrCount) {
  // Get debug location information
  DebugLoc DL;
  for (MachineInstr &MI : *Header) {
    if (!MI.isDebugInstr() && MI.getDebugLoc()) {
      DL = MI.getDebugLoc();
      break;
    }
  }

  LLVM_DEBUG({
    dbgs() << "Found qualifying loop in function " << MF.getName() << "\n";
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
    dbgs() << "  Instruction count: " << InstrCount << "\n";
    dbgs() << "  Loop header: BB#" << Header->getNumber() << "\n";
  });

#if 0
  // Insert NOP at the beginning of the loop
  MachineBasicBlock::iterator InsertPt = Header->begin();

  // Skip past any PHI nodes and debug instructions
  while (InsertPt != Header->end() &&
         (InsertPt->isPHI() || InsertPt->isDebugInstr())) {
    ++InsertPt;
  }

  // Insert NOP instruction using target-specific implementation
  TII->insertNoop(*Header, InsertPt);

  LLVM_DEBUG(dbgs() << "  Inserted NOP at the beginning of the loop\n");
#endif // if 0

  ++NumLoopsDetectedASM;

  const unsigned MachineWidth = 10;
  unsigned bubbles = InstrCount % MachineWidth;
  bubbles = bubbles ? MachineWidth - bubbles : 0;
  if (bubbles > MachineWidth / 3) {
    // Duplicate the loop body with unroll factor of 2
    duplicateLoopBody(Header, 2);
    ++NumLoopsUnrolledASM;
    return true;
  }

  return false;
}

unsigned LoopUnrollASM::countLoopInstructions(MachineLoop *Loop) {
  unsigned Count = 0;
  for (MachineBasicBlock *MBB : Loop->blocks()) {
    for (MachineInstr &MI : *MBB) {
      // Don't count debug instructions or pseudo instructions
      if (!MI.isDebugInstr() && !MI.isPseudo()) {
        ++Count;
      }
    }
  }
  return Count;
}

void LoopUnrollASM::duplicateLoopBody(MachineBasicBlock *Header,
                                      unsigned UnrollFactor) {
  assert (UnrollFactor > 1);

  MachineFunction *MF = Header->getParent();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  // Find the exit block (the fallthrough successor that's not the loop header)
  MachineBasicBlock *ExitBlock = nullptr;
  for (MachineBasicBlock *Succ : Header->successors()) {
    if (Succ != Header) {
      ExitBlock = Succ;
      break;
    }
  }

  if (!ExitBlock) {
    LLVM_DEBUG(dbgs() << "  Warning: Could not find loop exit block\n");
    return;
  }

  // Collect all non-terminator instructions to duplicate
  SmallVector<MachineInstr *, 16> InstsToClone;
  MachineInstr *OrigBranch = nullptr;

  for (MachineInstr &MI : *Header) {
    // Skip PHI nodes and debug instructions
    if (!MI.isPHI() && !MI.isDebugInstr()) {
      if (MI.isTerminator()) {
        // Find the conditional branch that goes back to the header
        if (MI.isConditionalBranch()) {
          for (MachineOperand &MO : MI.operands()) {
            if (MO.isMBB() && MO.getMBB() == Header) {
              OrigBranch = &MI;
              break;
            }
          }
        }
        // Don't add any terminators to InstsToClone
      } else {
        InstsToClone.push_back(&MI);
      }
    }
  }

  assert(OrigBranch && OrigBranch->isConditionalBranch() &&
         "No conditional branch found in loop");

  // Analyze the original branch to extract condition
  SmallVector<MachineOperand, 4> Cond;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;

  if (TII->analyzeBranch(*Header, TBB, FBB, Cond)) {
    LLVM_DEBUG(dbgs() << "  Warning: Could not analyze branch\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "  Original loop body has " << InstsToClone.size()
                    << " non-branch instructions\n");

  // We need to create new basic blocks for proper control flow
  // The structure will be:
  // Header -> Body1 -> Body2 -> ... -> BodyN -> Header (loop back)
  //    |        |        |              |
  //    v        v        v              v
  //  Exit    Exit      Exit           Exit

  // Create UnrollFactor-1 new basic blocks (we reuse Header for the first
  // iteration)
  SmallVector<MachineBasicBlock *, 4> NewBlocks;
  MachineBasicBlock *PrevBlock = Header;

  for (unsigned i = 1; i < UnrollFactor; ++i) {
    MachineBasicBlock *NewBB = MF->CreateMachineBasicBlock();
    MF->insert(++MachineFunction::iterator(PrevBlock), NewBB);
    NewBlocks.push_back(NewBB);
    PrevBlock = NewBB;
  }

  // Remove the original branch from Header - we'll add new ones
  TII->removeBranch(*Header);

  // Now set up each block with its body and appropriate branch
  MachineBasicBlock *CurrentBlock = Header;

  for (unsigned i = 0; i < UnrollFactor; ++i) {
    // For blocks after the first, copy the body instructions
    if (i > 0) {
      CurrentBlock = NewBlocks[i - 1];

      // Clone body instructions into the new block
      for (MachineInstr *MI : InstsToClone) {
        MachineInstr *ClonedMI = MF->CloneMachineInstr(MI);
        CurrentBlock->push_back(ClonedMI);
      }
    }

    // Insert appropriate branch for this iteration
    if (i < UnrollFactor - 1) {
      // For all but the last iteration, insert inverted conditional branch
      SmallVector<MachineOperand, 4> InvertedCond(Cond);
      bool Inverted = TII->reverseBranchCondition(InvertedCond);
      assert(!Inverted && "Unable to invert branch condition for conditional branch");
      (void)Inverted; // Silence unused variable warning in release builds

      // Next block in unrolled sequence
      MachineBasicBlock *NextBlock = (i == 0) ? NewBlocks[0] : NewBlocks[i];

      // Check if the next block is the layout successor (fallthrough)
      // If it is, we only need a conditional branch to exit
      if (CurrentBlock->isLayoutSuccessor(NextBlock)) {
        // Only insert conditional branch to exit, fall through to next block
        TII->insertBranch(*CurrentBlock, ExitBlock, nullptr, InvertedCond,
                          OrigBranch->getDebugLoc());
      } else {
        // Need both branches
        TII->insertBranch(*CurrentBlock, ExitBlock, NextBlock, InvertedCond,
                          OrigBranch->getDebugLoc());
      }
    } else {
      // For the last iteration, keep the original branch back to loop header
      // The exit block is typically the fallthrough, so we might only need
      // the conditional branch to Header
      if (CurrentBlock->isLayoutSuccessor(ExitBlock)) {
        // Only insert conditional branch to header, fall through to exit
        TII->insertBranch(*CurrentBlock, Header, nullptr, Cond,
                          OrigBranch->getDebugLoc());
      } else {
        // Need both branches
        TII->insertBranch(*CurrentBlock, Header, ExitBlock, Cond,
                          OrigBranch->getDebugLoc());
      }
    }
  }

  // Update the CFG - fix successor relationships
  // First update Header's successors
  if (UnrollFactor > 1 && !NewBlocks.empty()) {
    // Header now branches to either first new block or exit
    Header->removeSuccessor(Header); // Remove self-loop
    if (!Header->isSuccessor(NewBlocks[0]))
      Header->addSuccessor(NewBlocks[0]);
    // ExitBlock successor should already be there
  }

  // Now add successors for the new blocks
  for (unsigned i = 0; i < NewBlocks.size(); ++i) {
    MachineBasicBlock *BB = NewBlocks[i];
    // Each new block can exit
    if (!BB->isSuccessor(ExitBlock))
      BB->addSuccessor(ExitBlock);
    // And can continue to next block or loop back
    if (i < NewBlocks.size() - 1) {
      if (!BB->isSuccessor(NewBlocks[i + 1]))
        BB->addSuccessor(NewBlocks[i + 1]);
    } else {
      // Last block loops back to header
      if (!BB->isSuccessor(Header))
        BB->addSuccessor(Header);
    }
  }

  LLVM_DEBUG({
    dbgs() << "  Duplicated loop body with unroll factor " << UnrollFactor
           << "\n";
    dbgs() << "  Created " << NewBlocks.size() << " new basic blocks\n";
    dbgs() << "  Exit block: BB#" << ExitBlock->getNumber() << "\n";
  });
}

FunctionPass *llvm::createLoopUnrollASMPass() {
  return new LoopUnrollASM();
}
