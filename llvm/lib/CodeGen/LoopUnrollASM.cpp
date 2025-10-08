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
#include "llvm/ADT/Statistic.h"

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-asm"

STATISTIC(NumLoopsUnrolledASM, "Number of simple short loops detected by LoopUnrollASM");

static cl::opt<unsigned> LoopUnrollASMMaxInsts(
    "loop-unroll-asm-max-insts",
    cl::desc("Maximum number of instructions in a loop for LoopUnrollASM to process"),
    cl::init(50), cl::Hidden);

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
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool processLoop(MachineLoop *Loop, MachineFunction &MF);
  bool processTightLoop(MachineLoop *Loop, MachineFunction &MF,
                        MachineBasicBlock *Header, unsigned InstrCount);
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

  // Check if the backedge is a conditional branch
  MachineBasicBlock::iterator LastI = Header->getLastNonDebugInstr();

  if (LastI == Header->end()) {
    return Changed;
  }

  if (!LastI->isConditionalBranch()) {
    LLVM_DEBUG(dbgs() << "  Loop backedge is not a conditional branch");
    if (LastI->isBranch()) {
      LLVM_DEBUG(dbgs() << " (unconditional branch)");
    } else if (LastI->isTerminator()) {
      LLVM_DEBUG(dbgs() << " (non-branch terminator)");
    } else {
      LLVM_DEBUG(dbgs() << " (not a terminator)");
    }
    LLVM_DEBUG(dbgs() << "\n");
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
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Get debug location information
  DebugLoc DL;
  for (MachineInstr &MI : *Header) {
    if (!MI.isDebugInstr() && MI.getDebugLoc()) {
      DL = MI.getDebugLoc();
      break;
    }
  }

  // Print debug information
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

  // Update statistics
  ++NumLoopsUnrolledASM;

  return true;
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

FunctionPass *llvm::createLoopUnrollASMPass() {
  return new LoopUnrollASM();
}