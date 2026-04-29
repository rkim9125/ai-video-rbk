# Thesis 현황 카드 — 2026년 4월 29일 업데이트

**다음 supervisor 미팅:** 2026년 5월 6일 (수) — Nic
**Draft thesis 마감:** 2026년 5월 15일 (금) — Week 11
**Final thesis 마감:** 2026년 5월 29일 (금) — Week 13
**Presentation:** 2026년 6월 4–5일 (목–금) — Week 14

---

## 현재 상태

### 끝낸 것
- thesis_v6 작성됨: Chapter 1 (Introduction), Chapter 2 (Literature Review), Chapter 3 (Methodology)
- Experiment A–E 완료 (coarse GT 기준): representation, window size, boundary rule, structural signal, pruning distance — 5축 sensitivity sweep
- Coarse GT mismatch 진단 (lecture 2에서 qualitative error analysis)
- Experiment F: semi-automatic fine-grained hierarchical GT 구축 (coarse 대비 약 3.6배 dense)
- A–E를 fine GT 기준으로 재평가 — 최고 F1 = 0.408 (E2, min-distance 30s)
- 모든 CSV, lecture별 결과 정리 완료

### Outline만 있고 본문 안 쓴 것
- Chapter 4 (Results) — outline + 숫자 다 준비됨
- Chapter 5 (Discussion) — outline만
- Chapter 6 (Conclusion + Future Work) — outline만
- Abstract — 마지막에 작성

---

## Nic이 미팅에서 말한 것 (4/24)

| 지적 | 의미 |
|---|---|
| 글이 너무 fancy하고 읽기 어렵다 | 모든 문단 소리 내서 읽기. 자연스럽지 않으면 다시 쓰기. |
| F1 0.12 → 0.41이 3배라지만 결국 둘 다 0이다 | "Granularity 진단"만으로는 너무 방어적. **어떻게 풀 건지** path를 보여줘야 함. |
| Hierarchical이든 cross-lecture든 스토리에 맞는 걸로 골라라 | Nic이 결정 안 함 — 내가 판단해야 함. 다음 미팅까지. |
| Terminology 미리 정의하고 일관되게 써라 | Coarse GT, fine GT, boundary, clip-worthy 등 첫 등장 시 정의. |
| Diagram 더 넣어라 | 최소: coarse vs fine timeline + hierarchical pipeline + 실험별 schematic. |
| AI 사용은 OK, 다만 시간 절약될 때만 | AI text 그대로 붙이기 X. 본인 목소리로 다시 쓰기. |

---

## 내가 내린 결정

**Hierarchical coarse-then-fine prediction으로 간다.** Cross-lecture는 일단 보류.

**이유:** Nic이 가장 강하게 한 말이 *"안 되는 게 안 되는 거다, 어떻게 풀 거냐"*. Cross-lecture는 *"다른 강의에서도 똑같이 안 됨"* 만 보여줄 수 있어서 그 질문에 대답을 못 함. Hierarchical은 ceiling을 실제로 올리려는 시도라서 직접 대답이 됨. 현재 fine GT 기준 precision이 0.32니까 2-stage로 가면 false positive 줄어서 끌어올릴 여지가 있음.

Week 1에 시간 남으면 cross-lecture는 작은 부속 실험으로 추가.

---

## 새 thesis 스토리 (기존 framing 대체)

**기존 (방어적):** *"F1이 낮은 건 evaluation이 잘못됐기 때문. GT 바꾸면 3배 올라감."*

**새 (건설적):** *"Unsupervised + transcript-only 방식은 강의 데이터에서 F1 약 0.4가 현실적 ceiling이다. 이 ceiling이 어디서 오는지 진단했고, hierarchical 2-stage 구조가 그 ceiling을 어떻게 올리는지 보여준다. 거기서 production까지 가려면 [supervised fine-tuning / multimodal cue / human-in-the-loop] 가 필요하다 — future work에 명시."*

이 framing이 Abstract, Chapter 1, Chapter 5, Chapter 6 — 네 군데에 일관되게 흐르도록.

---

## 앞으로 3주 plan

### Week 1 — 4/24 ~ 5/1 (실험 + diagram)
- Hierarchical 2-stage prediction 구현 (coarse → 각 coarse segment 안에서 fine)
- 4개 강의 모두 돌리고 precision / recall / F1 계산 (fine GT 기준)
- Single-stage baseline (Exp E2) 와 비교
- 핵심 diagram 2개:
  - Coarse vs fine GT timeline (한 강의, 두 줄로 boundary 비교)
  - Hierarchical pipeline 아키텍처
- 시간 남으면 leave-one-lecture-out cross-validation 추가

### Week 2 — 5/2 ~ 5/8 (본문 작성)
- Chapter 4 Results: A–E coarse, F (fine GT 구축), A–E fine, hierarchical, optional cross-lecture
- Chapter 5 Discussion: ceiling 진단 + hierarchical lift + 솔직한 한계 + production까지 필요한 것
- 실험별 schematic (A, B, C, D) 추가 — 예쁘기보다 명확하게

### Week 3 — 5/9 ~ 5/15 (마무리 + 제출)
- Chapter 6 Conclusion + Future Work
- Abstract
- Ch 1–3 terminology pass (첫 등장 시 정의, 이후 일관성)
- 전체 thesis 소리 내서 읽기 (톤 점검)
- 5/15 금요일 draft 제출

### Draft 이후
- 5/15–5/29: Nic 피드백 반영 → final 제출
- 6/4–6/5: presentation (slide template 이미 project에 있음)

---

## 이번 주 액션 리스트 (4/24 → 4/30) — 다음 미팅 대비

### 수요일 미팅에 보여줄 것
1. **Hierarchical 2-stage prediction이 end-to-end로 작동** — 최소 1개 강의, 가능하면 4개 다. 결과가 엄청나야 하는 게 아니라 pipeline이 돌아야 함.
2. **결과 표** — single-stage (E2) vs hierarchical, fine GT 기준, 4개 강의, precision / recall / F1.
3. **Coarse vs fine GT timeline diagram** — thesis 전체의 headline figure.
4. **Hierarchical pipeline diagram** — 새 실험이 뭘 하는지 시각적으로.
5. **새 framing 한 단락 plain English** — 소리 내서 읽기 테스트 통과한 버전.

### 일별 목표
- **목 4/24 (오늘 저녁):** Hierarchical setup 결정 — coarse predictor는 Exp A baseline 재사용? Threshold는? Coarse segment 안에서 fine을 어떻게 돌릴지? 종이에 스케치.
- **금 4/25:** Coarse stage 구현 (기존 코드에서 min-distance만 늘려서 큰 section break 잡는 방식이 가장 빠를 듯).
- **토 4/26:** Fine stage를 각 coarse segment 안에서 돌리도록 구현. 두 단계 합치는 로직.
- **일 4/27:** 4개 강의 다 돌리고 CSV 생성, 숫자 sanity check.
- **월 4/28:** 핵심 diagram 2개 만들기. 결과 표 draft.
- **화 4/29:** Plain English 요약 한 단락 작성 + 미팅에서 Nic한테 walk through 할 짧은 script.
- **수 4/30:** 미팅.

### 이번 주에 안 할 것 (보류)
- Chapter 4 본문 작성
- Cross-lecture 실험 (hierarchical 일찍 끝나면 추가)
- Glossary 섹션
- Experiment A–D별 schematic
- Ch 1–3 terminology 다듬기

---

## 글쓰기 규칙 (지금부터 적용)

1. 모든 문단 소리 내서 읽기. 어색하면 다시 쓰기.
2. 한 문장 = 한 아이디어. 25 단어 넘으면 자르기.
3. **피하기:** *diagnostic contribution, systematically, paradigm, task-aligned, pedagogically salient, hierarchical (본문에서 — diagram에선 OK).*
4. **선호:** *this thesis shows, the method misses, the way we measure, a finer ground truth that matches what we want to clip.*
5. 새 용어는 첫 등장 시 정의.
6. AI output은 1차 draft로만. 최종 문장은 내 목소리로 직접.

---

## 수요일 미팅에 Nic한테 물어볼 것

1. Hierarchical 선택이 맞는지, 아니면 cross-lecture도 같이 원하는지?
2. 새 framing ("ceiling 인정 + 어떻게 올렸는지") 이 Nic이 원했던 방향인지?
3. Glossary 따로 안 만들고 본문에서 정의하는 걸로 OK인지?
4. Headline diagram (coarse vs fine timeline) 이 의도한 메시지를 잘 전달하는지?
5. 다음 checkpoint에서 보고 싶어 하는 건 뭔지?



1막 — Coarse GT 실험 (A–E 전부): "기존 방식대로 평가하면 F1이 0.12–0.17 ceiling에 갇힌다. 변수 다섯 개를 다 돌려도 그 이상 안 올라간다."
2막 — 진단 + Fine GT (Experiment F): "왜 안 올라가는지 봤더니 evaluation granularity가 task와 안 맞는다. Fine GT로 바꿔서 돌리면 같은 모델로 0.41까지 간다 — 즉 ceiling 일부는 모델 탓이 아니라 측정 탓이었다."
3막 — Hierarchical (새 실험): "Fine GT로도 ceiling이 0.41이다. 이건 진짜 모델의 한계다. 2-stage로 가면 그 한계가 어떻게 바뀌나?"

4.1 Experiments under coarse ground truth (Experiments A–E)
    ← 5개 실험 다 들어감, 표/숫자 그대로
    ← 결론: F1 ~0.12–0.17에서 ceiling

4.2 Diagnosing the coarse-GT ceiling (Experiment F)
    ← 왜 ceiling이 거기서 멈추는지
    ← Fine GT 구축 + 같은 모델 재평가
    ← 결론: F1 ~0.37–0.41로 상승, 그러나 새 ceiling 등장

4.3 Pushing the fine-GT ceiling (hierarchical 2-stage)
    ← 새 실험
    ← Single-stage vs 2-stage 비교
    ← 결론: F1 0.X로 상승 (or 못 올라가면 그것대로 finding)