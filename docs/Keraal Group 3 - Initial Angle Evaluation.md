---
title: Keraal Group 3 - Initial Angle Evaluation
aliases:
  - Keraal angle evaluation
  - Phone vs Vicon evaluation
tags:
  - fyp
  - physiotherapy
  - keraal
  - biomechanics
  - evaluation
status: preliminary
created: 2026-11-08
updated: 2026-11-08
---

# Keraal Group 3 - Initial Angle Evaluation

## Summary

Keraal Group 3 contains 540 labelled recordings with RGB video, BlazePose, Kinect, OpenPose and Vicon data. We loaded all 540 matched recordings and compared angles calculated from the supplied BlazePose skeleton against angles calculated from Vicon positions.

This is an **initial measurement result**, not the final validity result. The current comparison uses an approximate Vicon target mapping and a simple linear frame alignment. Large errors in some exercise/side combinations require coordinate and mapping validation before the numbers are used as final claims.

## Dataset

- **Dataset:** Keraal Group 3
- **Recordings:** 540 total
- **Exercises:** CTK, ELK and RTK
- **Recordings per exercise:** 180
- **Local path:** `data/keraal/`
- **Archive:** `data/keraal/group3.tar.xz`
- **Archive size:** 8,714,472,496 bytes
- **Extracted size:** approximately 18 GB
- **License:** CC BY-NC-SA, according to the [Keraal dataset page](http://nguyensmai.free.fr/KeraalDataset.html)
- **Data policy:** raw data is excluded from Git by `.gitignore`

The source format documentation is stored at `data/keraal/readme_files_format.txt`.

## Processing method

### BlazePose

1. Load the `positions` dictionary from the Keraal BlazePose JSON file.
2. Sort frames numerically.
3. Map the 33 landmarks to the repository's shared 13-joint layout.
4. Centre on the mid-hip and scale by median torso length.
5. Calculate joint angles using the shared angle definitions.

### Vicon

1. Load the 119 columns from the Vicon text file.
2. Interpret each target as 3D position plus quaternion.
3. Keep the 3D positions for the 17 targets.
4. Map Vicon targets to the shared 13-joint layout.
5. Resample Vicon positions to the BlazePose frame count using linear interpolation.
6. Centre on the mid-hip and scale by median torso length.
7. Calculate the same angle definitions as BlazePose.

Vicon uses the Z axis as the vertical axis. BlazePose image coordinates use an inverted Y axis. The angle engine therefore accepts an explicit `vertical_axis` parameter.

### Current angles

- Left elbow flexion
- Right elbow flexion
- Left shoulder abduction
- Right shoulder abduction
- Trunk lean

### Error metrics

For each recording and angle:

- Mean absolute error (MAE), in degrees
- Median absolute error, in degrees

The summary reports the mean and sample standard deviation of each per-recording metric grouped by exercise.

## Aggregate results

| Exercise | Recordings | Left elbow MAE | Right elbow MAE | Left shoulder abduction MAE | Right shoulder abduction MAE | Trunk lean MAE |
|---|---:|---:|---:|---:|---:|---:|
| CTK | 180 | 16.21° | 23.29° | 22.66° | 25.81° | 27.36° |
| ELK | 180 | 55.65° | 25.20° | 76.03° | 81.53° | 21.00° |
| RTK | 180 | 58.45° | 19.48° | 23.59° | 23.02° | 15.45° |

The complete machine-readable results are:

- [Per-recording metrics](../outputs/keraal_angle_per_recording.csv)
- [Exercise summary](../outputs/keraal_angle_summary.csv)

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/evaluate_keraal.py --root data/keraal
```

The evaluator writes:

```text
outputs/keraal_angle_per_recording.csv
outputs/keraal_angle_summary.csv
```

To evaluate only a small sample:

```bash
.venv/bin/python scripts/evaluate_keraal.py \
  --root data/keraal \
  --limit 3 \
  --output /tmp/keraal-per-recording.csv \
  --summary-output /tmp/keraal-summary.csv
```

The single-recording comparison command is:

```bash
.venv/bin/python scripts/compare_keraal.py \
  --root data/keraal \
  --index 0
```

## Interpretation

The CTK results are comparatively lower for elbow and shoulder angles. ELK and RTK show very large left-side errors for some angles. These results should not yet be described as definitive phone measurement accuracy.

Likely contributors include:

- Approximate correspondence between Vicon targets and anatomical joints
- Different coordinate systems and camera viewpoints
- BlazePose's monocular depth estimate versus Vicon's 3D positions
- Occlusion and visibility differences between exercise views
- Possible left/right or exercise-specific conventions in the source data
- Simple endpoint-based temporal resampling without lag correction

The trunk-lean result initially appeared incorrectly high because the Vicon vertical axis was treated as image Y. After checking the Vicon coordinates, the comparison was corrected to use Vicon Z as vertical.

## Kinect versus Vicon validation

Keraal Group 3 also provides Kinect 3D positions. We compared Kinect and Vicon using the same shared-joint mapping, angle definitions and frame resampling. This separates issues in the Vicon mapping from issues introduced by monocular BlazePose.

| Exercise | Kinect-Vicon left elbow MAE | Kinect-Vicon right elbow MAE | Kinect-Vicon left shoulder MAE | Kinect-Vicon right shoulder MAE | Kinect-Vicon trunk MAE |
|---|---:|---:|---:|---:|---:|
| CTK | 19.39° | 17.93° | 22.57° | 25.83° | 19.97° |
| ELK | 51.21° | 22.21° | 23.23° | 26.88° | 12.31° |
| RTK | 54.00° | 21.42° | 13.64° | 14.98° | 9.32° |

The complete comparison is available at [Kinect-Vicon per-recording metrics](../outputs/keraal_kinect_vicon_per_recording.csv).

### Finding

The large left-elbow errors in ELK and RTK are present in both Kinect-Vicon and BlazePose-Vicon comparisons. They are therefore not evidence that BlazePose alone caused those errors; the Vicon target mapping, exercise geometry or left-side convention needs further investigation.

The ELK shoulder result is different: Kinect-Vicon shoulder errors remain around 23–27°, while BlazePose-Vicon shoulder errors are approximately 76–82°. This points to a separate monocular BlazePose or visibility problem for ELK shoulder measurements.

## Temporal alignment diagnostic

We searched whole-frame shifts from -30 to +30 frames after resampling Vicon to the Kinect frame count. The shift was selected by minimizing the mean error across the five current angles.

- Recordings evaluated: 540
- Median best shift: -1 frame
- Median error improvement: 0.03°
- Recordings with any improvement: 391/540

The small median improvement means that whole-frame temporal lag is not the main explanation for the large ELK/RTK left-side errors. The per-recording results are available at [Kinect-Vicon alignment diagnostics](../outputs/keraal_kinect_vicon_alignment.csv).

## Vicon side-convention diagnostic

We also tested a diagnostic mapping that swaps every Vicon left/right target chain before comparing it with Kinect. This was used only to test a possible convention mismatch; the standard mapping remains unchanged.

| Exercise | Angle | Standard MAE | Swapped MAE | Change |
|---|---|---:|---:|---:|
| ELK | Left elbow | 51.2° | 26.8° | -24.5° |
| ELK | Right elbow | 22.2° | 54.9° | +32.7° |
| ELK | Left shoulder abduction | 23.2° | 77.9° | +54.7° |
| RTK | Left elbow | 54.0° | 22.8° | -31.2° |
| RTK | Right elbow | 21.4° | 55.7° | +34.3° |

The swap improves one side while worsening the other, and it severely worsens ELK shoulder measurements. It is therefore **not a valid global correction**. The evidence points to exercise-specific target geometry, target placement, or an angle-definition issue. The full per-recording diagnostic is available at [Vicon mapping diagnostics](../outputs/keraal_mapping_diagnostic.csv).

## Source-code mapping review

The Keraal authors' public MATLAB processing code was reviewed:

- [`loadData.m`](https://raw.githubusercontent.com/nguyensmai/KeraalDataset/main/evaluation_mouvements_gmm_matlab/functions/loadData.m) loads 25-joint Kinect positions and derives relative positions for anatomical Kinect joints.
- [`estimateOrientationFromPosition.m`](https://raw.githubusercontent.com/nguyensmai/KeraalDataset/main/evaluation_mouvements_gmm_matlab/functions/estimateOrientationFromPosition.m) derives orientations from the Kinect joint chains.
- [`displaySequence.m`](https://raw.githubusercontent.com/nguyensmai/KeraalDataset/main/evaluation_mouvements_gmm_matlab/functions/displaySequence.m) visualises the Kinect skeleton.
- The public processing code does not define a Vicon ART-target-to-Kinect-anatomical-joint conversion.

This means there is no source-backed mapping correction we can safely apply to the current Vicon comparison. The current mapping remains a clearly labelled approximation for diagnostics, not a validated ground-truth anatomical mapping. We should either obtain the authors' Vicon marker-placement/processing details or report Vicon results as target-based rather than clinical joint-angle validation.

## Kinect as the primary anatomical reference

Because the Kinect format provides anatomical joints and the authors' processing code explicitly uses those joints, the next measurement pass uses Kinect as the primary reference for phone-angle development. This does not make Kinect perfect: it is still a consumer depth sensor and has its own tracking noise. It does make the comparison semantically clearer than treating ART target positions as joint centres.

The phone sequence is mapped to the shared 13-joint layout, normalised, and resampled to the Kinect frame count. Kinect is mapped using the documented 25-joint order, normalised, and evaluated with an upward-positive Y axis. Both modalities then use the same five angle definitions.

Run the evaluation with:

```bash
.venv/bin/python scripts/evaluate_keraal_kinect.py --root data/keraal
```

Outputs:

- [BlazePose-Kinect per-recording metrics](../outputs/keraal_kinect_angle_per_recording.csv)
- [BlazePose-Kinect exercise summary](../outputs/keraal_kinect_angle_summary.csv)

### BlazePose versus Kinect results

| Exercise | Recordings | Left elbow MAE | Right elbow MAE | Left shoulder abduction MAE | Right shoulder abduction MAE | Trunk lean MAE |
|---|---:|---:|---:|---:|---:|---:|
| CTK | 180 | 26.63° | 29.75° | 19.57° | 19.66° | 24.09° |
| ELK | 180 | 23.65° | 24.24° | 79.84° | 82.27° | 19.97° |
| RTK | 180 | 27.81° | 27.32° | 22.44° | 20.65° | 20.27° |

These are the current primary measurement results. The summary CSV also contains per-recording standard deviations and median absolute errors.

### Finding

The Kinect comparison changes the interpretation of the earlier Vicon results:

- ELK and RTK elbow errors are now approximately 24–28° on both sides. The earlier 51–54° left-elbow errors were primarily caused by comparing Kinect anatomical joints with the approximate Vicon ART-target mapping.
- ELK shoulder abduction remains approximately 80° against Kinect, so this is a genuine problem in the current BlazePose angle definition, landmark visibility, depth estimate, or exercise geometry—not just a Vicon reference problem.
- CTK and RTK shoulder errors are approximately 20–22°, while trunk lean is approximately 20–24° across exercises.

The results support using Kinect for developing and debugging the phone measurement pipeline, while retaining Vicon as supplementary target-based evidence until its marker conversion is clarified.

## ELK shoulder-angle diagnostic

The ELK shoulder result was investigated without changing the production angle definitions. We compared three candidate definitions between BlazePose and Kinect:

1. **Torso-relative 3D:** the current angle between the side hip-to-shoulder vector and shoulder-to-elbow vector.
2. **Vertical 3D:** the angle between the shoulder-to-elbow vector and the vertical axis.
3. **Torso-relative 2D:** the angle between the mid-hip-to-shoulder vector and shoulder-to-elbow vector using only image-plane coordinates.

| Exercise | Side | Torso-relative 3D | Vertical 3D | Torso-relative 2D |
|---|---|---:|---:|---:|
| CTK | Left | 19.57° | 52.59° | 42.25° |
| CTK | Right | 19.66° | 51.67° | 42.31° |
| ELK | Left | 79.84° | 67.02° | 93.05° |
| ELK | Right | 82.27° | 64.79° | 90.89° |
| RTK | Left | 22.44° | 56.88° | 33.39° |
| RTK | Right | 20.65° | 55.10° | 36.87° |

The current torso-relative 3D definition remains the best of the tested candidates, but it is not reliable for ELK. The vertical and 2D alternatives are worse, so replacing the metric globally would reduce validity for CTK and RTK. The diagnostic output is available at [shoulder-angle diagnostics](../outputs/keraal_shoulder_angle_diagnostic.csv), and the code is [diagnose_shoulder_angles.py](../scripts/diagnose_shoulder_angles.py).

### Decision

Do not change the shared production angle definition yet. ELK should use an exercise-specific measurement design after inspecting the exercise view and landmark visibility. Candidate directions are:

- Use a different body landmark or plane for flank-stretch quality.
- Add landmark visibility/confidence gating and report “not reliable” when shoulder geometry is occluded.
- Define the ELK error detector from trunk tilt and arm alignment rather than a single shoulder-abduction scalar.

## Next validation tasks

- [x] Test a global Vicon left/right swap; it was rejected as a correction.
- [x] Review the authors' public visualization and processing code; it does not define a Vicon anatomical conversion.
- [x] Compare Vicon against Kinect 3D positions as an intermediate check.
- [x] Estimate residual whole-frame temporal lag; it was negligible overall.
- [x] Run BlazePose-vs-Kinect evaluation as the primary anatomical-reference pass.
- [x] Test alternative shoulder-angle definitions; no global replacement was justified.
- [ ] Request Vicon marker-placement/processing details from the Keraal authors.
- [ ] Reframe or replace the Vicon comparison if an authoritative conversion cannot be obtained.
- [ ] Run and record the BlazePose-vs-Kinect primary-reference evaluation.
- [ ] Define exercise-specific angle conventions for CTK, ELK and RTK.
- [ ] Inspect ELK video/landmark visibility and design an exercise-specific flank-stretch measurement.
- [ ] Add range-of-motion error per repetition.
- [ ] Add Bland-Altman limits and ICC after the angle definition is stable.
- [ ] Break results down by participant, error label and camera visibility.
- [ ] Validate the five angles on a small manually inspected set before reporting aggregate results.

## Related code

- [`src/rehabform/keraal.py`](../src/rehabform/keraal.py)
- [`src/rehabform/skeleton.py`](../src/rehabform/skeleton.py)
- [`scripts/compare_keraal.py`](../scripts/compare_keraal.py)
- [`scripts/evaluate_keraal.py`](../scripts/evaluate_keraal.py)
- [`data/README.md`](../data/README.md)

## Reproducibility checks

- 540/540 recordings had all five expected modalities.
- Core tests: 5/5 passing.
- Python compilation: passed.
- Dataset files and generated outputs remain excluded from Git.
