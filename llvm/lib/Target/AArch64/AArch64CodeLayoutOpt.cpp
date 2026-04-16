//===-- AArch64CodeLayoutOpt.cpp - Code Layout Optimizations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs after instruction scheduling and employs code layout
// optimizations for certain patterns.
//
// Option -aarch64-code-layout-opt selects instruction pairs to optimize:
//   fcmp-fcsel: Enable FCMP-FCSEL code layout optimization
//   cmp-csel:   Enable CMP/CMN-CSEL code layout optimization
//
// The initial implementation induces function alignment to help optimize
// code layout for the detected patterns.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-code-layout-opt"
#define AARCH64_CODE_LAYOUT_OPT_NAME "AArch64 Code Layout Optimization"

enum CodeLayoutOpt {
  FcmpFcsel, // FCMP-FCSEL code layout optimization (requires hasFuseFCmpFCSel)
  CmpCsel,   // CMP-CSEL code layout optimization (requires hasFuseCmpCSel)
  PageCross, // Page-cross alignment for large nested-loop functions
};

static cl::bits<CodeLayoutOpt> EnableCodeAlignment(
    "aarch64-code-layout-opt", cl::Hidden, cl::CommaSeparated,
    cl::desc("Enable code alignment optimization for instruction pairs"),
    cl::values(
        clEnumValN(FcmpFcsel, "fcmp-fcsel", "FCMP-FCSEL pair alignment"),
        clEnumValN(CmpCsel, "cmp-csel", "CMP/CMN-CSEL pair alignment (32-bit)"),
        clEnumValN(PageCross, "page-cross",
                   "Page-cross alignment for large nested-loop functions")));

static cl::opt<unsigned> FunctionAlignBytes(
    "aarch64-code-layout-opt-align-functions", cl::Hidden,
    cl::desc("Function alignment in bytes for code layout optimization "
             "(must be a power of 2)"),
    cl::init(64), cl::callback([](const unsigned &Val) {
      if (!isPowerOf2_32(Val))
        report_fatal_error(
            "aarch64-code-layout-opt-align must be a power of 2");
    }));

static cl::opt<unsigned> PageCrossMinInsns(
    "aarch64-code-layout-opt-page-cross-min-insns", cl::Hidden,
    cl::desc("Minimum instruction count for page-cross alignment"),
    cl::init(350));

static cl::opt<unsigned> PageCrossMinDepth(
    "aarch64-code-layout-opt-page-cross-min-depth", cl::Hidden,
    cl::desc("Minimum loop nest depth for page-cross alignment"), cl::init(2));

STATISTIC(NumFunctionsAligned,
          "Number of functions with aligned (to 64-bytes by default)");
STATISTIC(NumFcmpFcselPairsDetected,
          "Number of FCMP-FCSEL pairs detected for alignment");
STATISTIC(NumCmpCselPairsDetected,
          "Number of CMP/CMN-CSEL pairs detected for alignment");
STATISTIC(NumPageCrossAligned,
          "Number of functions page-aligned for page-cross optimization");

namespace {

class AArch64CodeLayoutOpt : public MachineFunctionPass {
public:
  static char ID;
  AArch64CodeLayoutOpt() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return AARCH64_CODE_LAYOUT_OPT_NAME;
  }

private:
  const AArch64InstrInfo *TII = nullptr;
  MachineLoopInfo *MLI = nullptr;

  // Returns true if MBB contains at least one layout-sensitive pattern.
  bool detectLayoutSensitivePattern(MachineBasicBlock *MBB);

  // Returns the page-cross alignment in bytes, or 0 if criteria not met.
  unsigned getPageCrossAlignment(MachineFunction &MF);

  bool optimizeForCodeLayout(MachineFunction &MF);
};

} // end anonymous namespace

char AArch64CodeLayoutOpt::ID = 0;

INITIALIZE_PASS(AArch64CodeLayoutOpt, "aarch64-code-layout-opt",
                AARCH64_CODE_LAYOUT_OPT_NAME, false, false)

void AArch64CodeLayoutOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createAArch64CodeLayoutOptPass() {
  return new AArch64CodeLayoutOpt();
}

/// Returns true if Opc is a floating-point comparison (FCMP/FCMPE).
static bool isFloatingPointCompare(unsigned Opc) {
  switch (Opc) {
  case AArch64::FCMPSrr:
  case AArch64::FCMPDrr:
  case AArch64::FCMPESrr:
  case AArch64::FCMPEDrr:
  case AArch64::FCMPHrr:
  case AArch64::FCMPEHrr:
    return true;
  default:
    return false;
  }
}

/// Returns true if Opc is a floating-point conditional select (FCSEL).
static bool isFloatingPointConditionalSelect(unsigned Opc) {
  switch (Opc) {
  case AArch64::FCSELSrrr:
  case AArch64::FCSELDrrr:
  case AArch64::FCSELHrrr:
    return true;
  default:
    return false;
  }
}

/// Returns true if MI is a qualifying 32-bit CMP or CMN instruction.
/// CMP is encoded as SUBS with WZR destination, CMN as ADDS with WZR.
/// Only simple variants (no shifted/extended reg) qualify, and immediate
/// variants require no LSL shift and small immediates (<=15).
static bool isQualifyingIntCompare(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AArch64::SUBSWrr:
  case AArch64::ADDSWrr:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr);
  case AArch64::SUBSWri:
  case AArch64::ADDSWri:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           MI.getOperand(3).getImm() == 0 && MI.getOperand(2).getImm() <= 15;
  case AArch64::SUBSWrs:
  case AArch64::ADDSWrs:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           !AArch64InstrInfo::hasShiftedReg(MI);
  case AArch64::SUBSWrx:
    return MI.definesRegister(AArch64::WZR, /*TRI=*/nullptr) &&
           !AArch64InstrInfo::hasExtendedReg(MI);
  default:
    return false;
  }
}

bool AArch64CodeLayoutOpt::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableCodeAlignment.getBits())
    return false;

  const Function &F = MF.getFunction();
  // hasOptSize() returns true for both -Os and -Oz.
  if (F.hasOptSize())
    return false;

  const auto *Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  bool HasPairOpt =
      (EnableCodeAlignment.isSet(FcmpFcsel) && Subtarget->hasFuseFCmpFCSel()) ||
      (EnableCodeAlignment.isSet(CmpCsel) && Subtarget->hasFuseCmpCSel());
  bool HasPageCross = EnableCodeAlignment.isSet(PageCross);

  if (!HasPairOpt && !HasPageCross)
    return false;

  return optimizeForCodeLayout(MF);
}

// Returns true if MBB contains at least one layout-sensitive pair.
// A pair is: a qualifying lead instruction immediately followed by its
// consumer (FCMP→FCSEL or CMP/CMN→CSEL), with no intervening instructions.
bool AArch64CodeLayoutOpt::detectLayoutSensitivePattern(
    MachineBasicBlock *MBB) {
  auto Instrs = instructionsWithoutDebug(MBB->begin(), MBB->end());
  auto End = MBB->instr_end();

  // --- FCMP-FCSEL detection ---
  if (EnableCodeAlignment.isSet(FcmpFcsel)) {
    if (llvm::any_of(Instrs, [End](MachineInstr &MI) {
          if (!isFloatingPointCompare(MI.getOpcode()))
            return false;
          auto NextIt =
              skipDebugInstructionsForward(std::next(MI.getIterator()), End);
          return NextIt != End &&
                 isFloatingPointConditionalSelect(NextIt->getOpcode());
        })) {
      ++NumFcmpFcselPairsDetected;
      return true;
    }
  }

  // --- CMP/CMN-CSEL detection ---
  if (EnableCodeAlignment.isSet(CmpCsel)) {
    if (llvm::any_of(Instrs, [End](MachineInstr &MI) {
          if (!isQualifyingIntCompare(MI))
            return false;
          auto NextIt =
              skipDebugInstructionsForward(std::next(MI.getIterator()), End);
          return NextIt != End && NextIt->getOpcode() == AArch64::CSELWr;
        })) {
      ++NumCmpCselPairsDetected;
      return true;
    }
  }

  return false;
}

/// Returns the maximum loop nest depth across all loops in the function.
static unsigned getMaxLoopDepth(MachineLoopInfo *MLI) {
  unsigned MaxDepth = 0;
  for (MachineLoop *L : *MLI)
    for (MachineLoop *Sub : L->getLoopsInPreorder())
      MaxDepth = std::max(MaxDepth, Sub->getLoopDepth());
  return MaxDepth;
}

unsigned AArch64CodeLayoutOpt::getPageCrossAlignment(MachineFunction &MF) {
  unsigned InsnCount = 0;
  unsigned SizeInBytes = 0;
  for (auto &MBB : MF)
    for (auto &MI : instructionsWithoutDebug(MBB.begin(), MBB.end())) {
      ++InsnCount;
      SizeInBytes += TII->getInstSizeInBytes(MI);
    }

  if (InsnCount <= PageCrossMinInsns || InsnCount >= 1100) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": page-cross: " << MF.getName()
                      << " instruction count " << InsnCount
                      << " outside range (" << PageCrossMinInsns
                      << ", 1100)\n");
    return 0;
  }

  unsigned MaxDepth = getMaxLoopDepth(MLI);
  if (MaxDepth < PageCrossMinDepth) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": page-cross: " << MF.getName()
                      << " max loop depth " << MaxDepth << " < "
                      << PageCrossMinDepth << "\n");
    return 0;
  }

  unsigned AlignBytes = NextPowerOf2(SizeInBytes - 1);
  LLVM_DEBUG(dbgs() << DEBUG_TYPE ": page-cross: " << MF.getName()
                    << " qualifies (insns=" << InsnCount
                    << ", size=" << SizeInBytes << ", depth=" << MaxDepth
                    << ", align=" << AlignBytes << ")\n");
  return AlignBytes;
}

bool AArch64CodeLayoutOpt::optimizeForCodeLayout(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << DEBUG_TYPE ": optimizeForCodeLayout: " << MF.getName()
                    << "\n");

  // Check page-cross alignment first (higher alignment takes precedence).
  if (EnableCodeAlignment.isSet(PageCross)) {
    unsigned AlignBytes = getPageCrossAlignment(MF);
    if (AlignBytes && MF.getAlignment() < Align(AlignBytes)) {
      MF.setAlignment(Align(AlignBytes));
      ++NumPageCrossAligned;
      LLVM_DEBUG(dbgs() << DEBUG_TYPE ": Set " << AlignBytes
                        << "-byte alignment for function " << MF.getName()
                        << "\n");
      return true;
    }
  }

  for (auto &MBB : MF) {
    if (!detectLayoutSensitivePattern(&MBB))
      continue;

    if (MF.getAlignment() >= Align(FunctionAlignBytes)) {
      LLVM_DEBUG(dbgs() << DEBUG_TYPE ": Function " << MF.getName()
                        << " already has sufficient alignment\n");
      return false;
    }

    MF.setAlignment(Align(FunctionAlignBytes));
    ++NumFunctionsAligned;
    LLVM_DEBUG(dbgs() << DEBUG_TYPE ": Set " << FunctionAlignBytes
                      << "-byte alignment for function " << MF.getName()
                      << "\n");
    return true;
  }

  return false;
}
