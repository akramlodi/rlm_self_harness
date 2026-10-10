# Exact test tasks and all attempts

Attempt numbers identify separate executions, not paired random seeds. F1 is the saved verifier score, with declared wrong-format failures scored zero.

## T03/W9

`oolong-t03-w9-39ef914e8722333f`; gold pair count: 406.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a01.json)|False|0.932|precision=0.873 recall=1.000 f1=0.932 missing=0 extra=59|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a02.json)|False|0.900|precision=0.819 recall=1.000 f1=0.900 missing=0 extra=90|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a03.json)|False|0.931|precision=0.931 recall=0.931 f1=0.931 missing=28 extra=28|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a01.json)|False|0.865|precision=0.865 recall=0.865 f1=0.865 missing=55 extra=55|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a02.json)|False|0.932|precision=0.873 recall=1.000 f1=0.932 missing=0 extra=59|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t03-w9-39ef914e8722333f__a03.json)|False|0.931|precision=0.931 recall=0.931 f1=0.931 missing=28 extra=28|

## T05/W10

`oolong-t05-w10-d04805539072461c`; gold pair count: 300.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or numeric value, and all instances that are an entity for both users must be before March 15, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a01.json)|False|0.960|precision=0.923 recall=1.000 f1=0.960 missing=0 extra=25|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a02.json)|False|0.960|precision=0.923 recall=1.000 f1=0.960 missing=0 extra=25|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a02.json)|False|0.958|precision=1.000 recall=0.920 f1=0.958 missing=24 extra=0|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t05-w10-d04805539072461c__a03.json)|False|0.843|precision=0.843 recall=0.843 f1=0.843 missing=47 extra=47|

## T06/W10

`oolong-t06-w10-9b441d42b516682e`; gold pair count: 903.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a location or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a02.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a03.json)|False|0.952|precision=1.000 recall=0.908 f1=0.952 missing=83 extra=0|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a02.json)|False|0.976|precision=1.000 recall=0.953 f1=0.976 missing=42 extra=0|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t06-w10-9b441d42b516682e__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|

## T07/W10

`oolong-t07-w10-9a0d3e39673a7350`; gold pair count: 120.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or numeric value, and all instances that are a numeric value for both users must be after February 1, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a02.json)|False|0.875|precision=0.875 recall=0.875 f1=0.875 missing=15 extra=15|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a01.json)|False|0.938|precision=0.882 recall=1.000 f1=0.938 missing=0 extra=16|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a02.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t07-w10-9a0d3e39673a7350__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|

## T11/W10

`oolong-t11-w10-68f90f0d94d504da`; gold pair count: 42.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity and one with abbreviation, and the other user has exactly one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a01.json)|False|0.511|precision=0.462 recall=0.571 f1=0.511 missing=18 extra=28|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a02.json)|False|0.381|precision=0.381 recall=0.381 f1=0.381 missing=26 extra=26|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a03.json)|False|0.631|precision=0.466 recall=0.976 f1=0.631 missing=1 extra=47|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a01.json)|False|0.914|precision=0.949 recall=0.881 f1=0.914 missing=5 extra=2|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a02.json)|False|0.000|no '(user_id_1, user_id_2)' pair and no 'No valid pairs' marker to parse|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t11-w10-68f90f0d94d504da__a03.json)|False|0.909|precision=1.000 recall=0.833 f1=0.909 missing=7 extra=0|

## T12/W9

`oolong-t12-w9-7fd1c6c04d9c0437`; gold pair count: 29.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a02.json)|False|0.906|precision=1.000 recall=0.828 f1=0.906 missing=5 extra=0|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a03.json)|False|0.125|precision=0.667 recall=0.069 f1=0.125 missing=27 extra=1|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a02.json)|False|0.000|no '(user_id_1, user_id_2)' pair and no 'No valid pairs' marker to parse|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w9-7fd1c6c04d9c0437__a03.json)|False|0.935|precision=0.879 recall=1.000 f1=0.935 missing=0 extra=4|

## T12/W10

`oolong-t12-w10-5787f2c8d5f82235`; gold pair count: 0.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a01.json)|False|0.000|precision=0.000 recall=0.000 f1=0.000 missing=0 extra=12|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a02.json)|False|0.000|precision=0.000 recall=0.000 f1=0.000 missing=0 extra=23|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a01.json)|False|0.000|precision=0.000 recall=0.000 f1=0.000 missing=0 extra=12|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a02.json)|False|0.000|precision=0.000 recall=0.000 f1=0.000 missing=0 extra=12|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t12-w10-5787f2c8d5f82235__a03.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|

## T13/W10

`oolong-t13-w10-a393091893c98516`; gold pair count: 37.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with description and abstract concept, and the other user has at least one instance with abbreviation and at least one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a01.json)|False|0.450|precision=0.338 recall=0.676 f1=0.450 missing=12 extra=49|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a02.json)|False|0.868|precision=0.846 recall=0.892 f1=0.868 missing=4 extra=6|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a03.json)|False|0.690|precision=0.617 recall=0.784 f1=0.690 missing=8 extra=18|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a01.json)|False|0.943|precision=1.000 recall=0.892 f1=0.943 missing=4 extra=0|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a02.json)|False|0.828|precision=0.720 recall=0.973 f1=0.828 missing=1 extra=14|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t13-w10-a393091893c98516__a03.json)|False|0.879|precision=1.000 recall=0.784 f1=0.879 missing=8 extra=0|

## T17/W9

`oolong-t17-w9-cc7785997a8db97f`; gold pair count: 116.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with numeric value, and the other user has at least one instance with location and at least one instance with description and abstract concept. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a01.json)|True|1.000|precision=1.000 recall=1.000 f1=1.000 missing=0 extra=0|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a02.json)|False|0.931|precision=0.884 recall=0.983 f1=0.931 missing=2 extra=15|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a03.json)|False|0.963|precision=0.928 recall=1.000 f1=0.963 missing=0 extra=9|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a01.json)|False|0.518|precision=0.815 recall=0.379 f1=0.518 missing=72 extra=10|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a02.json)|False|0.978|precision=1.000 recall=0.957 f1=0.978 missing=5 extra=0|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t17-w9-cc7785997a8db97f__a03.json)|False|0.894|precision=0.820 recall=0.983 f1=0.894 missing=2 extra=25|

## T20/W9

`oolong-t20-w9-d123b419c88623ed`; gold pair count: 21.

In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with numeric value and at least one instance with human being, and the other user has at least one instance with location, at least one instance with entity, and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.

|Harness|Attempt|Exact|F1|Verifier detail|
|---|---:|---|---:|---|
|initial|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a01.json)|False|0.950|precision=1.000 recall=0.905 f1=0.950 missing=2 extra=0|
|initial|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a02.json)|False|0.955|precision=0.913 recall=1.000 f1=0.955 missing=0 extra=2|
|initial|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/initial/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a03.json)|False|0.727|precision=0.696 recall=0.762 f1=0.727 missing=5 extra=7|
|sh_rlm|[1](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a01.json)|False|0.645|precision=1.000 recall=0.476 f1=0.645 missing=11 extra=0|
|sh_rlm|[2](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a02.json)|False|0.647|precision=0.846 recall=0.524 f1=0.647 missing=10 extra=2|
|sh_rlm|[3](/Users/williamstanford/RESEARCH/rlm_self_harness/experiment_oolong_pairs_dsv4f_20260928_174735_3rounds/eval/sh_rlm/oolong_pairs_short/round_00/runs/oolong-t20-w9-d123b419c88623ed__a03.json)|False|0.950|precision=1.000 recall=0.905 f1=0.950 missing=2 extra=0|
