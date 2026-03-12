; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=fuse-csel -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m3 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m4 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m5 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a78 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a710 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a715 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a720 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a725 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-x4 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-x925 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-n2 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-n3 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v1 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v2 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=neoverse-v3 -debug-only=machine-scheduler 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that the scheduling model has an edge between the SUBS and the CSEL.
; CHECK-LABEL: test_sub_cselw:%bb.0
; CHECK: SU(2):   %3:gpr32common = ADDWri %1:gpr32common, 7, 0
; CHECK: SU(3):   dead $wzr = SUBSWri %0:gpr32common, 13, 0, implicit-def $nzcv
; CHECK:   Successors:
; CHECK:     SU(4): Ord  Latency=0 Cluster
; CHECK: SU(4):   %5:gpr32 = CSELWr %0:gpr32common, %3:gpr32common, 0, implicit killed $nzcv
; CHECK:   Predecessors:
; CHECK:     SU(3): Ord  Latency=0 Cluster
; CHECK: SU(5):   $w0 = COPY %5:gpr32

; Verify 32-bit SUBSWri with imm>15 is NOT fused (no Cluster edge).
; CHECK-LABEL: test_sub_cselw_large_imm:%bb.0
; CHECK: SU(3):   dead $wzr = SUBSWri
; CHECK-NOT:  SU(4): Ord  Latency=0 Cluster

; Verify 64-bit SUBSXri->CSELXr is NOT fused (no Cluster edge).
; CHECK-LABEL: test_sub_cselx:%bb.0
; CHECK: SU(3):   dead $xzr = SUBSXri
; CHECK-NOT:  SU(4): Ord  Latency=0 Cluster

define i32 @test_sub_cselw(i32 %a0, i32 %a1, i32 %a2) {
entry:
  %v0 = sub i32 %a0, 13
  %cond = icmp eq i32 %v0, 0
  %v1 = add i32 %a1, 7
  %v2 = select i1 %cond, i32 %a0, i32 %v1
  ret i32 %v2

; CHECK-LABEL: test_sub_cselw:
; CHECK: cmp {{w[0-9]}}, #13
; CHECK-NEXT: csel {{w[0-9]}}
}

define i32 @test_sub_cselw_large_imm(i32 %a0, i32 %a1, i32 %a2) {
entry:
  %v0 = sub i32 %a0, 16
  %cond = icmp eq i32 %v0, 0
  %v1 = add i32 %a1, 7
  %v2 = select i1 %cond, i32 %a0, i32 %v1
  ret i32 %v2

; CHECK-LABEL: test_sub_cselw_large_imm:
; CHECK: cmp {{w[0-9]}}, #16
; CHECK-NEXT: csel {{w[0-9]}}
}

define i64 @test_sub_cselx(i64 %a0, i64 %a1, i64 %a2) {
entry:
  %v0 = sub i64 %a0, 13
  %cond = icmp eq i64 %v0, 0
  %v1 = add i64 %a1, 7
  %v2 = select i1 %cond, i64 %a0, i64 %v1
  ret i64 %v2

; CHECK-LABEL: test_sub_cselx:
; CHECK: cmp {{x[0-9]}}, #13
; CHECK-NEXT: csel {{x[0-9]}}
}
