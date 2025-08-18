//===-- ExpandPostRAPseudos.cpp - Pseudo instruction expansion pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that expands COPY and SUBREG_TO_REG pseudo
// instructions after register allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "asma"

namespace {
struct AsmAnalysis : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  AsmAnalysis() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    //AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// runOnMachineFunction - pass entry point
  bool runOnMachineFunction(MachineFunction &) override;
};
} // end anonymous namespace

char AsmAnalysis::ID = 0;
char &llvm::AsmAnalysisID = AsmAnalysis::ID;

INITIALIZE_PASS(AsmAnalysis, DEBUG_TYPE, "Pre-emit ASM analysis pass", false, false)

static void reportLoop(MachineLoop *L, const unsigned NestLevel,
                       MachineOptimizationRemarkEmitter *ORE) {
  static const unsigned MW=10;
  const MachineInstr *AnchorMI = nullptr;
  unsigned numInsts = 0;
  unsigned numSubLoops = 0;
  LLVM_DEBUG(dbgs() << ">> Loop=" << L->getStartLoc() << " Nest=" << NestLevel << "\n");

  for (auto *SubL: *L) {
    numSubLoops += 1;
    reportLoop(SubL, NestLevel+1, ORE);
  }

  for (auto &MI : *L->getHeader()) {
    if (MI.isDebugInstr() || MI.isPseudoProbe())
      continue;
    LLVM_DEBUG(dbgs() << ">>> " << MI);
    // Use the first instruction in the header with a DebugLoc
    if (!AnchorMI)
       AnchorMI = &MI;
    numInsts++;
  }

  if (MachineBasicBlock *Latch = L->getLoopLatch()) {
    for (MachineInstr &MI : Latch->terminators()) {
        if (MI.getDebugLoc()) {
            AnchorMI = &MI;
            break;
        }
    }
}
#if 0
  if (Loop *IRLoop = L->getLoopFor(L->getHeader()->getBasicBlock())) {
    DL = IRLoop->getStartLoc();
  }
#endif
  if (AnchorMI) {
    unsigned numBlocks = L->getNumBlocks();
    MachineOptimizationRemarkAnalysis R(DEBUG_TYPE, "LoopInfo2", 
        AnchorMI->getDebugLoc(), L->getHeader());
    R << "asm-loop with blocks=" << ore::NV("NumBlocks", numBlocks) <<
      " insts=" << ore::NV("NumInsts", numInsts) <<
      " bubbles=" << ore::NV("Bubbles", (numInsts % MW) ? MW - (numInsts % MW) : 0) <<
      " subLoops=" << ore::NV("NumSubLoops", numSubLoops);
    ORE->emit(R);
  }
}

/// runOnMachineFunction - Reduce subregister inserts and extracts to register
/// copies.
///
bool AsmAnalysis::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "> Machine Function: " << MF.getName() << '\n');
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  auto &ORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();

  for (auto *L: MLI) {
    reportLoop(L, 0, &ORE);
  }
  return false;

  unsigned NumFGs = 0;
  unsigned Sizes = 0;
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.pred_size() == 1 &&
        *std::prev((*MBB.pred_begin())->succ_end()) == &MBB) {
      // Single predecessors, do not start new fetch group.
      continue;
    }


    LLVM_DEBUG(dbgs() << "Starting new FG;");
    MachineBasicBlock *Curr = &MBB;
    unsigned Size = 0;
    SmallPtrSet<MachineBasicBlock *, 8> Seen;
    while (true) {
      if (!Seen.insert(Curr).second)
        break;

      Size += Curr->size();
      LLVM_DEBUG({
        dbgs() << "  ";
        Curr->printName(dbgs(), 1);
        dbgs() << ", ";
      });

      if (Curr->succ_size() == 0)
        break;

      Curr = *std::prev(Curr->succ_end());
    }
    LLVM_DEBUG(dbgs() << ": " << Size << "\n");
    NumFGs += divideCeil(Size, 16);
    Sizes += Size;
  }

  float AvgSize = double(Sizes) / double(NumFGs);
  LLVM_DEBUG(dbgs() << "Average FGs " << AvgSize << "\n");

  ORE.emit([&]() {
        return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "AsmA",
                MF.getFunction().getSubprogram(), &MF.front())
           << ore::NV("AverageSize", AvgSize);
  });
  return false;
}

MachineFunctionPass *llvm::createAsmAnalysis() {
  return new AsmAnalysis();
}
