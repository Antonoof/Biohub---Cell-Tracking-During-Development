# Biohub: план улучшений от подтверждённого baseline .957

Дата: 2026-09-09. Кодовая точка отсчёта: `37dd8c591a661dce46f64fa5d8e553dad2b1bebf`.

Это новый рабочий план, а не отчёт о выполненных экспериментах. Он уточняет старый `IMPROVMENT_PLAN.md` по текущему коду. При подготовке документа новые GPU-прогоны, обучение и измерение приростов не выполнялись.

**Подтверждено пользователем:** новый код уже использовал лучшие артефакты и получил на submission те же `.957`. Подключение bundle и проверку совместимости считаем завершёнными. Повторять старый notebook, стартовый golden или переобучение ради воспроизведения `.957` не нужно.

Как пользоваться: раздел 3 — приоритеты; 4–6 — локальная оценка, кэши и анализ уже работающего решения; 7–14 — направления экспериментов, а не обязательные переобучения; 15 — скорость; 16 — оценка приростов; 17 — последовательный backlog; 18 — условия принятия изменений и следующего submission.

## 1. Решение и главный фокус

**Весь лучший `.957` bundle оставляем baseline без переобучения:** P1, P2, Model C и division models, SourceCardinality, UniGRAFT, Motion, ownership, EdgeGRAFT, CandidateGRAFT и DeepCenter. Существующие runtime-only specialists тоже сохраняем.

Стартовая последовательность: **готовый `.957` → сохранённые или заново полученные inference-кэши → локальная оценка и ошибки → эксперименты без обучения → challenger только выбранной стадии**. Уже имеющиеся совместимые predictions/evidence/reports используем повторно. Новый прогон нужен только для отсутствующих результатов, не для повторного доказательства совместимости.

У baseline фиксируем также пороги, TTA, fusion, candidate population и graph-настройки. Изменения проверяем в отдельных configs, не переписывая baseline. Первые эксперименты могут менять пороги и graph logic при тех же весах. Новая модель обучается только под конкретную гипотезу; остальные артефакты остаются прежними. P1/P2/C не перетренировываем в рамках этого плана.

Три наиболее сильные гипотезы по потенциалу:

1. **Motion на предсказанной истории:** устранить разрыв между обучением на GT-истории и последовательным serving, затем улучшить конкуренцию кандидатов и локальную модель движения.
2. **Совместное исправление графа:** выбирать совместимый набор замен, восстановлений и делений, а не только хорошие по отдельности операции. Сначала локальные компоненты EdgeGRAFT/ownership, не переписывание всего решения.
3. **Temporal division repair:** видеть деление в нескольких кадрах, исправлять время события и находить отсутствующую дочь. Это гипотеза с самым большим исследовательским потолком, но и с большей ценой/риском.

Первая волна без обучения: CandidateGRAFT threshold replay, абляции cleanup/guards, параметры motion assignment и взаимодействие стадий. Следующая волна, только после выбора направления по ошибкам: CatBoost ownership, новый Motion или EdgeGRAFT challenger. Training bank создаётся для выбранной стадии, не для всей цепочки заранее.

**Порядок определяется измеренным запасом метрики.** Если oracle покажет, что большинство потерь связано с отсутствующими дочерьми, division поднимается выше motion. Если правильных кандидатов достаточно, в первую очередь улучшаем выбор и совместимость решений.

## 2. Как действительно устроен текущий pipeline

Номера train-модулей не задают порядок обработки графа. В частности, Motion `09` работает до division/EdgeGRAFT, а не после них.

```text
Изображения 3D+t
  → frozen P1 / P2 / Model C
  → детекции, association/fusion, native evidence, исходный граф / ILP
  → фильтр дальних и не соседних по времени рёбер
  → motion relinking: может заменить весь исходный набор рёбер
  → single-parent repair
  → закрытие однокадровых gaps + DeepCenter veto
  → safe divisions + DeepCenter veto
  → division decoder stack:
      Model C / DivisionGBM / SourceCardinality / UniGRAFT / live_v2 / ownership
  → защита ownership/division-контрактов
  → EdgeGRAFT + cleanup
  → CandidateGRAFT перед pruning
  → isolated/short-track filtering → node budget → coordinate smoothing
  → CandidateGRAFT после cleanup/smoothing
  → проверки графа → submission.csv
```

Точки истины: [graph pipeline](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/graph/pipeline.py), [сборка runtime](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/graph/upgrade.py), [serving config](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/configs/infer.yaml).

Уточнения к старому плану:

- CONTINUE vs DIVIDE уже реализован в OptionHead. Source/pair-разделение тоже уже есть; его не нужно изобретать заново.
- DivisionGBM участвует в runtime-стеке, в том числе в подготовке/оценке вариантов. Нельзя считать, что все эти модели не используются.
- В EdgeGRAFT gate уже есть transaction-utility supervision. Следующий шаг — качество её оценки, свежесть входного графа и совместное применение операций.
- Motion checkpoint выбирается через serving rollout. Однако train-ветка `video_rows` всё ещё строит историю через GT parent: хороший evaluator не устраняет mismatch обучающих признаков.
- Текущий DeepCenter trainer умеет обучаться на разных видео. Утверждение «всегда обучается на одном эмбрионе» нельзя переносить со старого артефакта на нынешний trainer.
- CandidateGRAFT вызывается дважды. Его порог и признаки надо оценивать вместе с промежуточным удалением узлов и smoothing.
- Некоторые train-конфиги указывают на исторические `data/...` банки; наличие trainer не означает наличие воспроизводимого актуального генератора всех его входов.

## 3. Приоритеты: куда вкладывать время

Здесь «потенциал» — исследовательская оценка по механике ошибки, не измеренный прирост и не прогноз public/private LB.

| Очередь | Направление | Потенциал | Стоимость | Когда начинать |
| --- | --- | --- | --- | --- |
| Уже выполнено | Новый код + лучшие артефакты → submission `.957` | Подтверждённый baseline | Повтор не нужен | Закрыто пользователем |
| 0 | Переиспользовать/дополучить inference-кэши, локальные метрики и stage errors | Основа выбора экспериментов | Зависит от уже сохранённых результатов | Сразу, без train и compatibility audit |
| 1A | CandidateGRAFT thresholds + cleanup/guard ablations | Ограниченный–средний, дешёвый | Низкая после replay | После baseline |
| 1B | Motion assignment/guards и взаимодействия стадий с текущими весами | Средний при ошибках выбора | Низкая–средняя | После соответствующего trace/replay |
| 1C | Ownership: CatBoost + конкуренция вариантов | Средний; выше при ownership FN | Низкая–средняя | После выбора этой гипотезы, подготовки её банка и folds |
| 2 | Motion: predicted-history training + serving features | Высокий при history/crossing errors | Средняя | После trace и oracle |
| 3 | EdgeGRAFT: свежий банк + metric-aware joint transactions | Высокий при конфликтующих исправлениях | Средняя–высокая | После prefix replay |
| 4A | Division: temporal reranking существующих пар | Высокий при selection/time errors | Средняя–высокая | После division error audit |
| 4B | Division: локализация пропущенной дочери/времени события | Наибольший research-потолок при candidate FN | Высокая | Только при достаточном oracle room |
| 5 | Gap repair + DeepCenter как оценщик конкретной операции | Средний–высокий при gap/node FN | Средняя | После repair-bank |
| 6 | CatBoost/RealMLP/TabM ensembles и финальная калибровка | Добавочный; сам по себе не гарантирует большой gain | Средняя | На удачных задачах и свежих банках |
| 7 | Unified Endpoint Repair / local structured decoder | Высокий, но наиболее рискованный | Высокая | После доказанной пользы локального joint decode |
| 8 | Graph corruption, pseudo-labels, SSL нового repair encoder | Усилитель направлений выше | Средняя–высокая | Когда реальные ошибки и labels уже измерены |
| Отложено | Перетренировка P1/P2/C, огромный detector HPO | Сейчас вне основного плана | Высокая | Только отдельным решением |

## 4. Фаза A — использовать готовый .957 для локальных экспериментов

### A1. Что уже сделано и что действительно нужно сейчас

- [x] Лучшие артефакты существуют и работают в новом коде: пользователь получил submission `.957`.
- [x] Совместимость для старта подтверждена пользователем; повторный старый-vs-новый inference не является задачей плана.
- [ ] Сохранить ссылку/ID успешного run, его config, code revision и bundle identity. Это точка сравнения будущих изменений, не новый аудит всех весов.
- [ ] Найти уже сохранённые train predictions, native evidence, stage graphs и локальные отчёты именно этого recipe. Переиспользовать подходящие; недостающие получить inference текущих моделей.
- [ ] Выбрать development/control movie lists и сохранить параметры scorer. `.957` на LB не подменяет локальный score на train panel; они не обязаны совпадать.
- [ ] Использовать уникальные run-id для новых экспериментов. Текущий `init_run` допускает существующую директорию: не затирать baseline.
- [ ] Frozen bundle и успешный config не изменять. Challenger configs/weights/reports складывать отдельно.

Исторические training banks не нужны для запуска лучших артефактов. Отсутствие банка в текущей рабочей копии **не блокирует** baseline, локальный scoring или эксперименты с текущими весами. Разметку/feature bank для обучения готовить только после выбора конкретного challenger.

### A2. Оценить сохранённые результаты; прогнать только недостающее

Все команды ниже выполняются из корня репозитория в уже рабочем окружении. Это примеры использования существующих entry points, не обязательный повтор выполненных run. Использовать config из успешного `.957` submission; приведённый `configs/infer.yaml` подходит, только если это тот самый recipe.

Если есть сохранённый **train** CSV, сначала оценить его. В примере путь соответствует run `baseline_smoke_v1`; заменить его на фактический сохранённый run. Test submission нельзя оценивать как train panel:

```bash
uv run python -m biohub.metrics.evaluate \
  --config configs/infer.yaml --panel smoke \
  --pred-source csv --pred-dir runs/baseline_smoke_v1/workdir/submission.csv \
  --require-complete --evaluation-level legacy_parity \
  --run-id eval_baseline_smoke_v1
```

Если train predictions/evidence отсутствуют, получить их текущим serving. Малый запуск нужен для проверки новых cache/snapshot hooks, если они добавлены, а не для повторной проверки совместимости bundle:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m biohub.infer.run \
  --config configs/infer.yaml --panel smoke \
  --movies-dir kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --gpu-workers 1 --run-id baseline_smoke_v1
```

После такого запуска использовать evaluate выше. Уже существующие run-id не переиспользовать. Если нет промежуточных stage snapshots, добавить нужные hooks из C1 **до** дорогого полного inference, чтобы не повторять его только ради кэшей.

Только если полного train175 inference ещё нет:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 uv run python -m biohub.infer.run \
  --config configs/infer.yaml --panel train175 \
  --movies-dir kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --gpu-workers 5 --run-id baseline_train175_v1

uv run python -m biohub.metrics.evaluate \
  --config configs/infer.yaml --panel train175 \
  --pred-source csv --pred-dir runs/baseline_train175_v1/workdir/submission.csv \
  --require-complete --evaluation-level legacy_parity \
  --run-id eval_baseline_train175_v1
```

Важно: `--panel train175` выбирает ID, но не переключает корень изображений с test на train. Для train-прогонов всегда явно задавать `--movies-dir .../train`.

Для `held20` и `practice4` тоже сначала искать сохранённые результаты и дополнять только недостающие. `smoke` совпадает с practice4; повторный независимый score от этого не появляется. Панели `all199` сейчас нет: полный отчёт по 199 нужно собрать через API с явным movie list либо добавить отдельную панель. Не усреднять три panel scores — агрегировать исходные per-movie counts.

`--gpu-workers 5` управляет detector-процессами, но не делает пять DeepCenter workers: их число регулируется отдельно. В стартовом прогоне сохраняем graph-настройки baseline; throughput-профиль меняем отдельно.

Сравнение двух eval-run доступно как Python API, не как готовый CLI:

```bash
uv run python -c 'from pathlib import Path; import json; from biohub.validation.compare import compare_runs; print(json.dumps(compare_runs(Path("runs/eval_baseline_smoke_v1"), Path("runs/eval_candidate_smoke_v1")), indent=2))'
```

Последняя команда — шаблон: второй eval-run предварительно должен существовать; набор видео и evaluation level должны совпадать.

### A3. Что должно быть под рукой

Для локального baseline собрать из существующих run либо сохранить при недостающем inference:

- `submission.csv`, hashes, node/edge counts, `resolved_config.yaml`, `manifest.json`;
- `evaluation/per_movie.json`, `summary.json`, `completeness.json`;
- raw graph, native evidence P1/P2/C, финальные графы и stage timing;
- число full/combined, fallback/emergency, skipped/failed movies, а также причины;
- wall time, RAM/VRAM peak, размер кэшей, распределение времени по видео.

Полный CSV ещё не доказывает полный прогон нужной модели: submission может быть собран с fallback/emergency. Такие видео отдельно отмечать и не смешивать с качеством normal serving незаметно.

**Gate A:** подтверждённый `.957` bundle используется без изменений; на выбранной локальной панели есть полный score и необходимые predictions/evidence. Это подготовка к измерению улучшений, не повторная сертификация baseline и не обучение моделей.

## 5. Фаза B — валидация, которой можно доверять

### B1. Что оптимизируем

Текущий локальный scorer:

```text
J_edge(movie) = TP / (TP + FP + FN)
J_adj(movie) = max(0, J_edge(movie) * (1 - 0.1 * (N_pred - N_est) / N_est))
J_adj_all = weighted mean J_adj(movie), weight = TP + FP + FN
J_div_all = sum(div_TP) / sum(div_TP + div_FP + div_FN)
S = J_adj_all + 0.1 * J_div_all
```

Если division union равен нулю, код возвращает только adjusted edge component. Источник: [aggregation](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/metrics/aggregation.py).

Это не `0.9 × edge + 0.1 × division`, не среднее per-movie score и не обязательно score, ограниченный единицей. Поправка на число узлов в локальном коде двусторонняя. Её соответствие scoring server нельзя доказать локальным golden или прежними двумя одинаковыми округлёнными LB.

Для каждого эксперимента показывать `S`, обе компоненты, raw edge Jaccard, edge/division TP/FP/FN, node recall и node ratio. Не строить стратегию вокруг удаления узлов ради спорного множителя. Подтверждать рост реальных связей/делений и отдельно отслеживать влияние node penalty.

### B2. Честно назвать уровень оценки

- `legacy_parity`: воспроизведение frozen pipeline и его исторических артефактов.
- `conditional_head_oof`: новые головы обучаются с held-out видео, но работают поверх неизменного upstream, который мог видеть эти видео при своём обучении.
- `strict_nested`: только при выполненном внешнем исключении данных для всей заявленной обучаемой цепочки; не достигается переименованием manifest.

P2 по историческому описанию обучался на всех 199 видео. Поэтому с frozen P2 мы не получаем полностью независимую end-to-end OOF-оценку. Frozen downstream-модели также могут иметь историю обучения/настройки на этих видео. Это ограничение нужно записать для каждой зависимости, а не только для детектора.

Цель сейчас — надёжная **парная оценка новых downstream-изменений при фиксированном upstream**, с явно указанными ограничениями переноса на private test.

Это не требование переучить старые модели для «правильного baseline». Их качество на submission уже подтверждено; ограничения train-оценки просто отражаем в отчётах. На этапе без обучения фиксируем development/control panels и выбираем thresholds только на development, не называя настройку поверх frozen моделей новым OOF-обучением.

### B3. Folds для будущего challenger, не переобучение baseline

Следующие пункты применяются, **когда выбран эксперимент с обучением**. Для inference/threshold replay готового `.957` собирать новую OOF-цепочку всех моделей не требуется.

- [ ] Создать одну сохранённую `movie_id → outer_fold` карту для новых downstream-моделей. Проверить баланс эмбрионов, размеров видео и известных division events; учитывать связанные/перекрывающиеся видео как один group, если такие найдены.
- [ ] Не использовать разные `GroupKFold`, SHA256 и CRC32 mappings как будто это одни folds. У frozen runtime сохранить старую маршрутизацию; для нового artifact писать явный fold manifest.
- [ ] В каждом outer fold: обучение, feature selection, HPO, calibration, thresholds и blend weights используют только outer-train, с inner split/crossfit.
- [ ] Если переобучаем несколько зависимых стадий, внешнее validation-видео исключается из всех новых upstream heads, которые генерируют его входы. Для обучения следующей головы внутри outer-train нужны inner out-of-fold predictions предыдущей.
- [ ] Простая цепочка из независимо подготовленных OOF-таблиц не гарантирует nested validation: upstream-модель, создавшая train-строки следующей головы, могла обучаться на её outer-validation видео.
- [ ] Сохранять `fit_movie_ids`, `predict_movie_ids`, `parent_artifact_hashes` для каждого fold; тестировать их непересечение по всей заявленной обучаемой цепочке.
- [ ] Для финального refit на всех разрешённых train-данных использовать уже зафиксированные настройки. Это отдельный deploy artifact, не OOF artifact.

Существующий [compare_runs](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/validation/compare.py) проверяет уровни, movie sets и status. Расширить его проверками scorer/data/fold/upstream fingerprints и finite metrics. Сам флаг `evaluation_level` пока не доказывает provenance.

### B4. Control panel и статистика

1. Smoke — проверка работоспособности, не выбор победителя.
2. Для вариантов без обучения — фиксированный development split для настройки и отдельный control для проверки. Для нового обучаемого challenger — outer CV, с выбором настроек внутри outer-train.
3. Отдельный lockbox для редких проверок принятых изменений. Исторический held20 не называть untouched, если на нём уже выбирались старые модели/пороги; он остаётся полезным regression panel.
4. Embryo-swap для новых голов — stress test переноса, а не гарантия качества на новом эмбрионе; frozen upstream всё ещё тот же.
5. Показывать paired bootstrap по видео с пересчётом полной агрегации на каждом resample, per-embryo результаты, worst movies и gain concentration. Два эмбриона недостаточны для надёжной статистики межэмбрионального переноса.
6. Не считать сотни тысяч рёбер независимыми объектами для confidence interval. Не выбирать лучший из 200 trials по outer/lockbox score.

**Gate B для старта:** одинаковые scorer/movie lists для парных сравнений; development/control роли зафиксированы; известный bias frozen артефактов указан. **Дополнительный gate перед train:** происхождение features и изоляция labels подтверждены для новой обучаемой цепочки.

## 6. Фаза C — быстрый stage replay и EDA ошибок

### C1. Минимальная доработка инфраструктуры

Не начинать с тотального рефакторинга. Уже работающий новый serving оставляем источником истины. Добавить только то, чего не хватает для ближайшего эксперимента: сохранение нужного prefix, replay его суффикса и сравнение score. Готовые результаты использовать без повторного detector inference.

| Что сделать | Где изменить существующий код | Новый результат |
| --- | --- | --- |
| Snapshot hooks до/после graph-стадий | `modules/graph/pipeline.py`, `upgrade.py` | Графы, provenance узлов/рёбер, guards, timing |
| Replay с заданного prefix | `infer/run.py`, graph API | Повтор суффикса без нового detector inference |
| Bank builder только выбранной для обучения стадии, позже | Соответствующие `train/*`, `features/*`, runtime modules | Feature/label bank конкретного challenger |
| Единый experiment bundle / path overrides | `infer/config.py`, `utils/runs.py` | Безопасная подмена одной модели |
| Парный отчёт и oracle audit | `validation/compare.py`, `metrics/*`, `train/edgegraft_oracle.py` | Полный Δscore и error budget |

Предлагаемые **новые, пока не существующие** модули: `biohub.experiments.replay`, `biohub.experiments.build_bank`, `biohub.experiments.audit`, `biohub.experiments.search`, `biohub.experiments.export_bundle`. Это целевые интерфейсы; команд `python -m ...` для них пока нет.

Не реализовывать весь этот набор до первого эксперимента. Достаточно тонкого replay-wrapper вокруг действующих serving-функций. Training runner/bank builder/export расширяются только при необходимости обучения или замены артефакта.

В `build_tracking` / `load_tracking_config` уже есть программный `bundle_paths` override. Использовать его как основу, добавив явный проверяемый путь из experiment config/CLI. Простое добавление произвольного нового YAML-ключа не гарантирует, что текущий inference подхватит новые веса. До реализации override можно собрать отдельный bundle с ожидаемой структурой и указать его через существующий `bundle_dir`.

Snapshot — не только список рёбер. Сохранять raw attributes/probabilities, связь native IDs с graph IDs, injected-node provenance, guards/ownership contracts и настройки runtime. Stateful стадии и история motion должны воспроизводиться из того же prefix. Простого запуска постпроцессинга поверх уже обработанного final graph недостаточно.

Если новый вариант добавляет узлы, меняет координаты или кандидатов, нельзя без проверки переиспользовать старые matching/features. Нужно инвалидировать соответствующие зависимости и пересчитать их.

### C2. Схема кэшей и банков

Различать **inference-кэш** (предсказания готовых моделей, probabilities, графы) и **training bank** (признаки, labels, folds для обучения новой модели). Первый нужен для быстрых экспериментов без обучения; второй — только выбранному challenger. Разметка ошибок для отчёта сама по себе не требует запуска trainer.

Предлагаемая структура, создаваемая по мере реализации:

```text
artifacts/<baseline_id>/manifest.json
artifacts/<baseline_id>/native/{p1,p2,model_c}/<movie>.npz
artifacts/<baseline_id>/graphs/<stage>/<movie>.npz
artifacts/<baseline_id>/traces/<movie>.parquet
artifacts/<baseline_id>/evaluation/<stage>/...
data/banks/<stage>/<bank_hash>/{features,labels,manifest,...}
configs/experiments/<experiment_id>.yaml
runs/<experiment_id>/{models,oof,replay,evaluation,reports,...}
```

В manifest банка: версия генератора, content hashes данных/evidence/prefix graph, модель и config upstream, feature names/order/dtypes/units, label version, supervision mask, folds, полный movie list. Публикация через staging целиком; не смешивать старые и новые shard-файлы.

Разделить GT-labeling и serving features физически и логически. `movie_id` — ключ группировки, не признак для запоминания labels. GT match/parent/event ID, GT estimated node count и oracle decisions не подаются как serving-признаки.

Ниже — отложенная проверка входов **той стадии, которую решили обучать**. Существующие downstream YAML — спецификации trainer, не перечень обязательных стартовых запусков. Проверять по необходимости:

- `09_motion.yaml`: proposal export, splits и cache;
- `02_division*.yaml`, `03_cardinality.yaml`, `04_unigraft.yaml`: audit/event/full-population caches и legacy decoder paths; в decoder config есть абсолютный `/mnt/c/...` путь;
- `06_ownership.yaml`: ownership geometry bank;
- `07_edgegraft*.yaml`: labels, decisions, baseline graph и exact report должны происходить из одного prefix;
- `08_candidategraft*.yaml`: population и OOF report должны соответствовать одной модели/схеме/выборке, а не механически прикладываться к новому fit.

Справочник для будущих train-экспериментов, **не инструкция запустить все trainers**. После выбора одной стадии и подготовки её банка можно использовать соответствующую точку входа. Таблица показывает текущие trainers, не уже реализованные CatBoost/temporal/joint-decoder расширения. Создать отдельный YAML с актуальными входами и новым output:

| Стадия | Уже существующая точка запуска | Базовый YAML |
| --- | --- | --- |
| Motion residual | `uv run python -m biohub.train.09_motion --config <experiment.yaml>` | `configs/09_motion.yaml` |
| Ownership | `uv run python -m biohub.train.06_ownership --config <experiment.yaml>` | `configs/06_ownership.yaml` |
| EdgeGRAFT ranker | `uv run python -m biohub.train.07_edgegraft_ranker --config <experiment.yaml>` | `configs/07_edgegraft_ranker.yaml` |
| EdgeGRAFT gate | `uv run python -m biohub.train.07_edgegraft_gate --config <experiment.yaml>` | `configs/07_edgegraft_gate.yaml` |
| Daughter-pair MLP | `uv run python -m biohub.train.02_division --config <experiment.yaml>` | `configs/02_division.yaml` |
| Division decoder | `uv run python -m biohub.train.02_division_decoder --config <experiment.yaml>` | `configs/02_division_decoder.yaml` |
| SourceCardinality | `uv run python -m biohub.train.03_cardinality --config <experiment.yaml>` | `configs/03_cardinality.yaml` |
| UniGRAFT | `uv run python -m biohub.train.04_unigraft --config <experiment.yaml>` | `configs/04_unigraft.yaml` |
| CandidateGRAFT final fit | `uv run python -m biohub.train.08_candidategraft --config <experiment.yaml>` | `configs/08_candidategraft.yaml` |
| DeepCenter | `uv run python -m biohub.train.10_deepcenter --config <experiment.yaml>` | `configs/10_deepcenter.yaml` |

`09_motion_cache` — не то же самое, что Motion residual: это finetuning detector edge-head. Его не включаем в первый цикл с замороженными P1/P2/C. CandidateGRAFT final fit запускать только после отдельного OOF screen/selection, не вместо него.

### C3. Какие отчёты построить до крупных трейнов

По видео и по стадиям:

- Количество известных TP/FP/FN, division errors, изменение node matching, добавленные/удалённые узлы.
- Сколько правильных связей стадия исправила и сколько сломала; сколько её улучшений отменили следующие стадии.
- Ошибки по плотности, z-положению, времени, яркости/SNR, скорости/ускорению, disagreement P1/P2/C, близости к границе кадра/видео.
- Division FN: нет parent / нет дочери / есть обе, но нет пары / пара отвергнута / неверное время / daughter ownership conflict / guard / удаление cleanup.
- Link FN: нет узла / нет candidate edge / проигрыш правильного кандидата / неверная history / gap / конфликт / позднее удаление.
- FP отдельно от unknown. Отсутствие sparse GT не является отрицательным label.

Собрать фиксированный error atlas: крупнейшие потери по метрике + случайные ошибки + корректные трудные примеры обоих эмбрионов. Визуально проверять несколько кадров вокруг события, а не один кадр. Development atlas не должен включать скрытые labels lockbox для ручной настройки.

### C4. Три oracle, которые решают, куда идти

1. **Existing-candidate oracle:** какое качество достижимо выбором текущих допустимых вариантов без новых узлов? Это проверка selection bottleneck.
2. **Expanded-candidate oracle:** расширить label-free candidates — radius/top-k/native evidence/repair proposals — и измерить дополнительный достижимый gain. GT использовать для оценки вариантов, не для их генерации.
3. **Joint-decision oracle:** оценить совместимый набор операций вместо независимого выбора. Сохранить ограничения графа и отдельно прогнать итоговый суффикс pipeline.

Отдельная GT-injection диагностика может показать потери от отсутствующих узлов, но это заведомо нереалистичный сценарий, не serving-решение и не обещанный attainable gain. При sparse GT такой анализ ограничен аннотированной областью.

Если oracle использует greedy/эвристику, его результат — найденный diagnostic score, не доказанный математический upper bound. Oracle gains стадий не складываются. Для node/coordinate changes полный matching пересчитывается.

База уже есть: [EdgeGRAFT oracle](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/edgegraft_oracle.py), [component logic](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/edgegraft/component.py). Адаптировать к нынешним schemas/prefix; не считать его готовым oracle для всей цепочки.

**Gate C для первых экспериментов:** новый replay-wrapper не меняет baseline на том же кэше, есть stage deltas и первичная классификация ошибок. Это проверка нового инструмента replay, не повторная проверка уже работающего inference/bundle. Не ждать реализации всех трёх oracle: expanded/joint диагностику углублять перед соответствующим дорогим проектом.

### C5. Первая волна — вообще без обучения

Все перечисленные эксперименты используют текущие лучшие веса. Вначале меняем одну группу настроек за раз и запускаем нужный suffix, затем проверяем полезные сочетания.

1. **N1 — CandidateGRAFT:** пороги, pre-only/final-only/both, их взаимодействие с pruning. Сохранить model scores один раз, если входные features не меняются; переигрывать реальные graph decisions.
2. **N2 — cleanup:** short-track filtering, boundary rescue, smoothing, node budget. Измерять исправленные/сломанные edges и divisions, не только node count.
3. **N3 — текущий Motion:** residual strength, geometry/learned bonus, assignment cost caps и правила конкуренции. Историю для изменённого recipe переигрывать последовательно, не оценивать на frozen истории другого варианта.
4. **N4 — division/ownership/EdgeGRAFT guards:** пороги и разрешённые конфликты, shadow proposals, сохранение полезных операций следующими стадиями. Не отключать все protections одновременно.
5. **N5 — DeepCenter/gaps:** пороги текущего veto и геометрические gates, без нового image model.

Если параметр хранится в deploy report, а не YAML, менять только экспериментальную копию отчёта/артефакта; baseline bundle не редактировать. После успешных локальных абляций — небольшая совместная настройка на development и проверка control. Лишь затем выбирать, какая ошибка требует новых features, candidates или обучения.

## 7. Motion: кандидат на первый основной train после анализа ошибок

Код: [train/motion.py](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/motion.py), [graph/motion.py](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/graph/motion.py), [features/motion.py](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/features/motion.py), [config](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/configs/09_motion.yaml).

Текущий Motion checkpoint уже входит в `.957` и не требует переобучения для старта. Сначала N3 с ним. Эксперименты M0–M5 ниже — новые challengers при подтверждённых history/assignment errors; это не восстановление утраченной модели.

### Главные ограничения

- Train GT-history и serving predicted-history различаются. Ошибка одного шага меняет признаки следующих шагов.
- Train feature/candidate recipe использует собственные knobs (`tight=6.2`, `relaxed=9.5`, scalar velocity), тогда как serving использует другую настройку, axis weights и multi-step history. Нужно устранить расхождения семантически, не просто скопировать три числа.
- Независимый binary edge loss не полностью соответствует конкуренции за target и опции «не соединять».
- Небольшой MLP — не доказательство слабости. Без корректной истории более мощная модель может лучше выучить неверное распределение.

### Эксперименты в порядке выполнения

1. **M0 — только исправление train distribution.** Генерировать train traces тем же serving rollout с frozen corrector; GT применять только для labels/valid masks. Сохранить текущую MLP, loss и candidate population, насколько возможно.
2. Вынести общий serving feature builder и проверить равенство train/infer vectors на одной и той же predicted history. Learned edge probabilities обязательны; geometry-only — отдельно помеченная диагностика.
3. **M1 — refresh rollout.** После первого нового corrector пересобрать outer-train traces с inner-OOF моделями и повторить обучение. Outer-validation labels не участвуют в refresh/early stopping/HPO.
4. **M2 — competition-aware loss.** Группы source→targets + `NO_LINK`; добавить target competition/assignment-aware surrogate. `NO_LINK` размечать только при достаточной GT-supervision; unknown не превращать в отрицательный класс.
5. **M3 — новые признаки.** Past/future consistency, margin до второго кандидата, неоднозначность соседей, uncertainty истории, disagreement моделей. Существующие density/velocity/registration features не выдавать за новые.
6. **M4 — локальное поле движения.** Robust flow по уверенным соседним трекам и отклонение клетки от него; сравнить с текущим global registration. Исключить сам проверяемый edge из построения его подтверждающего сигнала и сохранять GT-free процедуру.
7. **M5 — смена модели.** CatBoost residual/ranker, более широкая MLP, затем TabM/RealMLP. Calibration/scale residual и assignment проверять совместно: классификационная probability не равна стоимости движения автоматически.

Основной выбор — полный serving rollout и затем весь downstream suffix. Дополнительно смотреть crossings, recovery после ошибки истории, TP/FP, orphan rate, candidate recall и latency на плотных кадрах.

**Где возможен прорыв:** правильная история и совместное assignment могут улучшать длинные последовательности, а не отдельные edges. Если existing-candidate oracle почти не лучше baseline, расширение MLP бессмысленно — работать с отсутствующими candidates/узлами.

## 8. Ownership: быстрый сильный challenger

Код: [trainer](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/ownership.py), [runtime](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/ownership.py), [config](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/configs/06_ownership.yaml).

Текущая модель — ExtraTrees: 400 деревьев, depth 3, около 15 serving features. Frozen report содержит 2,089,875 candidate rows, но порядка 37.7 тыс. известных train-строк; миллионы кандидатов не равны миллионам labels. На unseen movie runtime выбирает одну fold-модель по hash.

1. **O0 — без train:** использовать готовый ExtraTrees artifact как reference; собрать его решения, scores и ошибки на текущем графе, проверить threshold/guard hypotheses. Переобучение ExtraTrees не является условием старта.
2. **O1 — выбранный train-эксперимент:** подготовить только ownership bank и обучить CatBoost. Сравнить с frozen ownership по полному replay. Если нужно отдельно измерить эффект семейства модели, дополнительно обучить ExtraTrees на тех же features/labels/folds; это необязательный matched-training control, а не замена baseline.
3. **O2:** оценивать все допустимые parent/pair варианты и `KEEP_CURRENT`, не только бинарную корректность уже выбранной best pair. Добавить future persistence/divergence, motion consistency, конкурентных родителей и модельный disagreement.
4. **O3:** ранжировать конкурирующие варианты внутри события/компоненты; gate по ожидаемой полезности всей замены. CatBoost поддерживает pairwise/groupwise ranking objectives — [официальная документация](https://catboost.ai/docs/en/concepts/loss-functions-ranking). Их преимущество здесь требуется проверить экспериментально.
5. **O4:** seed ensemble, затем TabM/RealMLP при дополнительных правильных решениях. Blend/threshold fitting только внутри training folds.

Для OOF нельзя усреднять пять fold-моделей, четыре из которых видели данное видео. На truly unseen test усреднение возможно, но такой serving policy должен быть проверен на соответствующей held-out схеме, а не подменять прежний single-fold routing незаметно.

**Где возможен прорыв:** вернуть пропущенные корректные division ownership, не крадя дочерей у правильных родителей. AUC и source precision — диагностики; принимаем только выигрыш final score после всех guards и cleanup.

## 9. EdgeGRAFT: от хороших отдельных замен к хорошему графу

Код: [ranker](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/edgegraft_ranker.py), [transaction gate](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/edgegraft_gate.py), [runtime](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/edgegraft/runtime.py).

Frozen metric report: 97,484 transactions, 3,883 non-neutral labeled cases, 2,329 positive / 1,554 negative. У выбранной исторической feature family 2,057 wins и 206 losses. Это не свежий final replay; `metric_delta_sum=4.2969` — сумма локальных изменений, не прирост LB.

1. **E0 — текущие артефакты.** Сначала replay/аудит frozen ranker и gate: где хорошие transactions отклонены, конфликтуют или отменяются. Пороги и conflict resolution можно проверять без переобучения.
2. **E1 — выбранный model challenger.** Только после выбора этой гипотезы подготовить банк на prefix непосредственно перед EdgeGRAFT; переиспользовать подходящие candidates/evidence, не смешивать exact counts разных baseline. Обучить CatBoost и сравнить с frozen runtime; HGB на том же новом банке — дополнительный контроль семейства модели, если нужен.
3. **E2 — utility.** Сравнить бинарный знак полезности с magnitude-aware regression/ranking или отдельными heads ожидаемых ΔTP/ΔFP/ΔFN. Существующая локальная utility уже есть; задача — лучше предсказывать её и учитывать итоговую агрегацию.
4. **E3 — conflicts.** Строить graph of transactions: shared source/target, удаление ребра другой операции, division/ownership constraints. Включать `KEEP` и выбирать совместимые операции внутри малых компонент.
5. **E4 — local exact/beam solver.** Для простых one-to-one случаев — assignment; для fork/node transactions — ограниченный local solver. Ограничить размер/время, для больших компонент оставить проверенный fallback. Сравнить с имеющейся component-логикой, не писать её дубликат.
6. **E5 — iterative repair.** После принятого пакета пересчитать затронутые признаки; второй проход разрешать только при дополнительном inner-CV gain. Защититься от циклических замен.

При изменении нескольких операций нельзя считать сумму независимых deltas точной полезностью нового графа. Graph repair взаимодействует с matching, division counts и downstream pruning. Финальный replay обязателен.

**Где возможен прорыв:** убрать selection ceiling от локального greedy, получить пользу от согласованных перестановок родителей и перестать ломать правильные структуры ради отдельного уверенного edge.

## 10. Division stack: главный дорогой проект

Код: [division training](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/division.py), [decoder training](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/decoder.py), [cardinality](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/cardinality.py), [cardinality runtime](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/cardinality/graph.py), [live_v2 transactions](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/live_v2/transactions.py).

### D0. Сначала понять, где теряются события

Измерить source recall, наличие обеих дочерей, pair coverage, правильность времени, pair ranking, theft/conflict и позднее удаление. Декомпозиция из C3 должна объяснять FN в scoring semantics, а не только человеческую картинку.

Для DivisionGBM, SourceCardinality, UniGRAFT, live_v2 сделать отдельные абляции **с неизменными весами остальных стадий** и полный replay. Затем несколько наиболее важных парных абляций: отдельный модуль может казаться бесполезным, потому что его решения дублируются или отменяются.

### D1. Быстрые улучшения существующих голов

- Свежие source/pair banks из текущего prefix, согласованные folds и GT-known masks.
- CatBoost для source/pair GBM как challenger; OptionHead оставить baseline.
- Более сильный OptionHead/listwise ranking с нормализованной группой `CONTINUE + PAIRS`; temporal и disagreement признаки.
- Отдельная event confidence, не зависящая исключительно от наличия удачной пары. Событие без пары должно инициировать поиск дочери, а не исчезать из обучения.
- Калибровка cap/threshold/guard с учётом стоимости FP divisions и конкурирующего ownership.

### D2. Temporal reranker без изменения детекций

Новая модель видит crop вокруг parent в `t−2…t+3`, траекторию parent, варианты дочерей, native evidence/выборочно кэшированные признаки frozen P1/P2/C, плотность и геометрию.

Первая версия только ранжирует уже существующие пары и время события. Это отделяет value temporal encoder от сложности новых node proposals.

Обучение: event loss + permutation-invariant pair/trajectory supervision, hard negatives реальных соседних/пересекающихся клеток, маски доступности кадров и sparse labels. Нельзя штрафовать неаннотированную дочь как background.

### D3. Найти отсутствующую дочь и правильное время

Расширить outputs:

1. `CONTINUE / DIVIDE / INSUFFICIENT_EVIDENCE` либо калиброванная event uncertainty;
2. временное смещение division event;
3. две неупорядоченные daughter positions/offsets на `t+1`;
4. подтверждение траекторий/позиций на `t+2` и при наличии `t+3`;
5. варианты `reuse_existing_node` vs `propose_missing_node` с duplicate suppression.

Две дочери обучать с permutation-invariant assignment; учитывать физические единицы и anisotropic voxels. На границах видео — явная маска, не синтетический «будущий кадр».

Новые узлы добавлять только как часть проверяемой local transaction: parent→daughters→future support, с конкуренцией за существующие nodes, ограничениями indegree/outdegree и fallback на прежний граф. Проверять итоговый sparse scorer и node penalty; не считать красивую локализацию достаточным доказательством gain.

### D4. Что делать с UniGRAFT и live_v2

UniGRAFT оставить отдельным P1/P2-view specialist до доказательства, что объединение лучше. Его disagreement с Model-C view полезен для meta-gate.

live_v2 — runtime specialist, не отдельный полноценный trainer. Сначала измерить какие типы ошибок он исправляет/создаёт; экспортировать его кандидаты и причины принятия. Затем сравнить learned meta-gate над предложениями специалистов с текущими жёсткими guards. Старые специалисты удаляются только после парной абляции, а не ради красоты архитектуры.

**Где возможен прорыв:** добавить события, которые невозможно исправить простой сменой классификатора, потому что правильной дочери/пары/времени нет среди текущих вариантов. Если oracle показывает именно это, D2/D3 важнее широкого tabular HPO.

## 11. CandidateGRAFT: сначала раскрыть дешёвый запас

Код: [trainer](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/candidategraft.py), [runtime](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/candidategraft/runtime.py).

Текущий runtime: 8 features, native P1/P2 edges, только уже существующие узлы, `t→t+1`, свободный source и свободный target. Не умеет восстановить отсутствующую клетку и не заменяет занятое ошибочное ребро.

Исторический report: 29,231 кандидатов, но лишь 348 known rows: 304 positive / 44 negative. На этих OOF-строках threshold `0.9` давал 261 TP / 18 FP, а `0.5` — 293 TP / 19 FP. Это повод для replay, **не разрешение сразу поставить 0.5 в production**: популяция, конкуренция, неизвестные labels и два вызова меняют результат.

1. **C0 — без train:** sweep порогов готовой модели на фиксированном development split; полный replay обоих вызовов и всех промежуточных cleanup стадий, затем control. Frozen scores не называть новыми OOF, если соответствующая изоляция не подтверждена.
2. **C1:** абляция pre-only / final-only / both. Раздельные пороги или stage feature — только после проверки и inner tuning.
3. **C2:** добавить history/motion residual, relative rank/margin, future persistence, density, признаки endpoint и local conflict. Использовать ту же общую feature реализацию, что в serving.
4. **C3:** расширить label-free candidate union для существующих узлов. Это меняет training population и требует новой разметки/калибровки.
5. **C4 — только при выборе обучения:** подготовить bank этого challenger, graph corruption на train folds и OOF оценку новой модели; большой neural ensemble только при достаточном реальном signal и complementary errors.

Новый serving `final fit` нельзя оценивать на тех же known rows. Для OOF нужен fold-aware inference/export; нынешнее прикладывание screen-report к единственной final-fit модели не делает её OOF.

**Где возможен прирост:** быстро вернуть уверенные пропущенные continuations. Структурный потолок ограничен существующими свободными endpoints — missing-node/division задачи переносить в соответствующий repair-модуль.

## 12. Gap closing и DeepCenter

Код: [gaps](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/modules/graph/gaps.py), [DeepCenter trainer](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/src/biohub/train/deepcenter.py), [config](/Users/tochkamac/Documents/GitHub/Biohub---Cell-Tracking-During-Development/configs/10_deepcenter.yaml).

### Gaps

1. Посчитать сколько верных gaps отсутствует из-за radius, существующих-node conflicts, veto и лимитов synthetic nodes.
2. Сравнить существующий геометрический выбор с learned gap-transaction scorer: endpoint history + image support + будущая траектория.
3. `reuse_existing` должен конкурировать с `add_node` и `keep_gap`, а не автоматически проигрывать новому node.
4. После успеха на single-frame gap — двухкадровые gaps как отдельная гипотеза. Не добавлять длинное ребро через кадры в обход output contract: проверять цепочку допустимых узлов/рёбер.
5. Для нового узла сначала local image refinement и duplicate suppression, затем exact final evaluation.

### DeepCenter

1. Сначала threshold sweep текущего veto отдельно для gap и division, с одинаковым cached image evidence.
2. Построить **repair acceptance bank**: вход — конкретная предлагаемая операция и локальное 3D+t окружение; target — её польза/вред в графе на known supervision.
3. Проверить небольшой temporal crop encoder или head поверх frozen DeepCenter features. Не обязательно переобучать full-frame heatmap network для улучшения gate.
4. Затем сравнить обновлённый full-frame DeepCenter на групповых splits, если error atlas показывает потерю image signal именно в его heatmaps.
5. Выбор checkpoint: repair-gate utility и final graph score; heatmap loss и peak recall остаются диагностиками.

Проверить fail-open случаи при отсутствии heatmap, единицы координат/stride и кэш-ключи. Для нового checkpoint обновлять artifact metadata/epoch contract явно; не отключать проверки совместимости ради загрузки.

**Где возможен прорыв:** новые корректные узлы и links, для которых frozen detector дал недостаточный score, но последовательность изображений подтверждает клетку.

## 13. Association, ILP, cleanup и границы freeze

### Association / ILP при фиксированных P1/P2/C

- Сначала измерить, какие решения raw graph переживают motion, и через какие raw probabilities/candidates влияют на него. Замена raw edges не означает полного отсутствия косвенного влияния ILP.
- Абляции: ILP on/off, небольшой набор association blend/temperature/retention knobs, forward/backward evidence. Каждый такой эксперимент — новая версия входного графа/evidence и зависимых банков.
- Расширение top-k/radius делать по candidate-recall oracle и memory/time budget, не максимизировать число кандидатов само по себе.
- Если cache содержит только уже отфильтрованные probabilities, из него нельзя честно восстановить другой upstream threshold/fusion. Сохранить достаточный raw evidence либо повторить frozen inference.
- Переписывать ILP на C++ только если профилирование показывает значимую долю общего времени. Сам язык реализации не обещает прироста метрики.

### Cleanup / smoothing

- По отдельности и совместно проверить short-track filtering, boundary rescue, node budget, isolated pruning, smoothing и два CandidateGRAFT вызова.
- Отмечать, какие истинные links/divisions уничтожены каждым правилом; сохранять причину удаления.
- Smoothing способен менять node matching даже без изменения topology. Его оценивать полным scorer, в том числе около division и быстрых движений.
- Если жёсткий min track length теряет важные короткие треки, сравнить track-quality gate с image/motion/temporal support. Не обучать его на «нет GT значит FP».
- Guard relaxation — отдельный эксперимент: сначала shadow proposals, потом final replay с ограниченным конфликтным solver. Не выключать все protection contracts одновременно.

### Что делать с неудачным detector train

P1/P2/C не перетренировываем. Разрешён один короткий диагностический шаг: old weights и новый checkpoint через один evaluator, один movie list, одинаковые threshold/matching и одинаковую метрику. Это выяснит, сравнивались ли `0.9` и `0.4` вообще в одной шкале. Не превращать эту проверку в новый цикл detector Optuna.

Если позднее oracle покажет непреодолимый missing-node ceiling, вернуться к вопросу отдельного repair detector или к P1/P2/C только по новому решению.

## 14. Глобальные улучшения после первых побед

### 14.1. Unified Endpoint Repair

Общий label-free proposal interface для `KEEP`, `LINK`, `REPLACE_PARENT`, `CLOSE_GAP`, `DIVIDE`, `ADD_SUPPORTED_NODE`. Общие temporal/image/motion признаки, но type-specific heads и thresholds.

Сначала объединить **формат кандидатов, scoring и conflict resolution**, сохранив существующие генераторы. Лишь затем проверять одну большую модель или graph transformer/GNN. Модель получает локальное 3D+t окружение и сообщения между конкурирующими операциями; decoder обеспечивает временную направленность, cardinality и atomic updates.

Переход к global solver оправдан, если local-component oracle показывает существенную потерю от дальних взаимозависимостей. Не запускать огромный full-video ILP/GNN без такого доказательства.

### 14.2. Graph corruption

Генерировать на outer-train и только в достоверно размеченных локальных структурах: удалить edge, поменять parent, закрыть правильный endpoint ложным, удалить промежуточный node, украсть дочь, сдвинуть division time.

Частоты и сложность брать из реальных ошибок baseline. Смешивать с реальными ошибками и сохранять source type. Corruptions одного видео не должны попадать в разные folds. Валидация всегда на естественных, не синтетически испорченных graphs.

### 14.3. Pseudo-labels

Сначала высокоуверенные continuations с согласованностью P1/P2, motion и future evidence. Divisions псевдоразмечать гораздо осторожнее. Confidence weights, отдельный supervision mode, сравнение с GT-only и teacher-bias audit обязательны.

Teacher для outer-validation не использует его labels. Pseudo-labels не заменяют ground truth в валидации и не превращают unlabeled в negative. Использование test images/transductive training или внешних данных — только после отдельной проверки актуальных правил соревнования; по умолчанию здесь их не используем.

### 14.4. SSL для нового repair encoder

Если temporal division/gap encoder ограничен labels, сравнить pretraining на train images: temporal consistency, masked reconstruction локальных crops, correspondence. P1/P2/C остаются frozen. Учитывать время pretraining и доказывать downstream gain над тем же encoder без SSL.

### 14.5. Ensemble

Не `CatBoost + RealMLP + TabM везде` по умолчанию. Сначала сильный single model, затем анализ несовпадающих правильных/ошибочных **graph decisions**. Усреднение AUC-скореров с одинаковыми ошибками может не менять final graph вообще.

TabM реализует parameter-efficient ensemble MLP — [официальный репозиторий](https://github.com/yandex-research/tabm). Это подходящий challenger, не доказательство преимущества в нашей задаче. Для маленького CandidateGRAFT с 44 историческими negatives важнее labels/validation; для богатых ownership/EdgeGRAFT/motion banks проверка neural diversity содержательнее.

Blending делать по совместимым scores/costs с inner calibration. Структурный ensemble — объединение graph proposals и learned selection — сравнить с простым усреднением вероятностей. Финальный ансамбль обязан помещаться в Kaggle inference budget.

## 15. Скорость экспериментов и использование 5 H200

Главный ускоритель этого плана — не повторять тяжёлый detector на каждом downstream trial.

| Участок | Что сделать | Как доказать ускорение без потери качества |
| --- | --- | --- |
| Frozen P1/P2/C inference | Один evidence run на фиксированный recipe; shard по видео на 5 GPU | Полное покрытие и стабильные hashes/выходы |
| Graph replay | Повторять только изменившийся suffix; кэшировать неизменные heatmaps/features | Replay baseline равен full baseline |
| Feature banks | Разделить extraction и fit; contiguous float32 arrays, per-movie partition, memory map где оправдан | Cold/hot wall time, RAM, равенство features |
| Geometry | Spatial index/radius queries вместо полных distance matrices, переиспользование neighborhoods | Полнота candidates, особенно boundary cases |
| Tabular train | CPU/GPU benchmark на реальном банке; предсказания батчами, без per-row pandas loops | Полное fit+predict+replay время, не только fit |
| Small MLP | Данные в RAM/VRAM при разумном объёме, большие векторизованные batches | Samples/s, отсутствие CPU launch bottleneck |
| Temporal crop train | Cached crop indices, локальное хранилище, grouped reads, bounded prefetch; features только нужных ROI | Loader wait/H2D/forward/backward timings |
| Model execution | BF16/compile/checkpointing как отдельные измеряемые опции на H200 | Warm-up отдельно; numerical/metric parity, peak VRAM |
| Local decoder | Независимые conflict components, bounded exact solver + fallback | Runtime tail и same/better graph score |
| Evaluation | Кэш GT, process-level per-movie scoring, избегать повторной загрузки | Полный scorer output совпадает |

На 5 H200 сначала один процесс/trial или fold на GPU. Для маленькой модели 5-GPU DDP может проиграть пяти независимым экспериментам. Graph CPU workers, BLAS/OpenMP threads и GPU workers ограничивать совместно: `n_jobs=-1` в каждом из пяти trials создаёт oversubscription.

DeepCenter не размножать в десятках процессов на одной GPU. Его кэш и batch service должны соответствовать наличию GPU и лимиту памяти; текущие worker caps сохранять до отдельного stress test.

Не держать все full-resolution feature maps P1/P2/C по всем фильмам без расчёта размера. Для temporal heads начать с ROI features/pooled embeddings, а logits/native evidence хранить один раз. Сжатие/precision кэшей — отдельный speed-quality tradeoff.

### HPO без бессмысленных 200 полных прогонов

В первой волне это tuning graph-параметров **без обучения**, на inference-кэшах. Model/loss search и training runner из пунктов ниже нужны только выбранному обучаемому challenger. В обоих случаях подбор параметров отделять от контрольной оценки.

1. Smoke матрица поддерживаемых сочетаний: модель/loss/device/features/supervision/decoder. Невалидные combinations отбрасывать до загрузки данных/GPU.
2. Сначала несколько осмысленных recipes и ablations; не смешивать новую задачу, банк, модель и decoder в одном search space.
3. Дешёвый screening на inner folds/фиксированном development subset. Proxy допустим только после проверки связи с final score.
4. Лучшие кандидаты — полный inner replay и несколько seeds; выбранный вариант — outer assessment.
5. 200 trials оправданы после доказанного signal и измеренной цены trial. Не «оптимизировать» outer/lockbox 200 раз.
6. Study хранит data/code/config/fold hashes, runtime budget, причины FAIL/PRUNED; после несовместимых изменений — новая study. `FAIL` не превращается в score `0` и не исчезает из отчёта.

Текущий detector-specific search нельзя считать готовым generic search всех downstream стадий. При необходимости общий runner реализовать и протестировать отдельно; это не условие первого ручного threshold/guard replay.

## 16. Как оценивать возможный прирост без выдуманных обещаний

Baseline `.957` уже подтверждён submission. Но до локальных stage errors и oracle **численные прогнозы приростов по стадиям недостоверны**. Не обещаем `.957 → .97/.98` и не складываем гипотетические gains. Получение локального error budget не требует обучения новой baseline-модели.

Точная связь при наличии division component:

| Изменение компоненты при прочих равных | Изменение общего S |
| --- | --- |
| adjusted edge Jaccard +0.001 | +0.001 |
| adjusted edge Jaccard +0.005 | +0.005 |
| division Jaccard +0.01 | +0.001 |
| division Jaccard +0.05 | +0.005 |
| division Jaccard +0.10 | +0.010 |

Division исправление также меняет edge component, а новые узлы меняют matching/node penalty — поэтому реальный полный gain считать scorer, а не этой таблицей по одной части.

Стартовые **пороги практической ценности**, а не прогноз результатов:

- Дешёвый threshold/feature эксперимент: интересен устойчивый gain порядка `+0.0005…+0.001`, если он выше измеренной нестабильности и не держится на одной аномалии.
- Новый tabular/MLP модуль: разумная цель `+0.001…+0.003` на полной парной оценке при небольшом serving cost. Меньший устойчивый gain тоже можно принять, если цена почти нулевая.
- Дорогой temporal division/structured decoder: запускать, когда diagnostic oracle показывает запас хотя бы нескольких тысячных или больше; это основание инвестировать, не обещание взять весь запас.

Для каждой идеи вести карточку:

```text
experiment_id / hypothesis / one primary change
baseline + prefix + dataset + bank + fold hashes
evaluation level и известные источники bias
candidate coverage / diagnostic oracle room
inner selection protocol / chosen hyperparameters
outer ΔS, Δedge, Δdivision, TP/FP/FN, node statistics
per-embryo, worst movies, paired uncertainty, gain concentration
latency/RAM/VRAM, fallback counts
decision: accept / reject / investigate
```

Идея становится «прорывом», только когда добавляет существенный **новый** выигрыш поверх уже принятых стадий, переносится между development splits и проходит контрольный serving-прогон.

## 17. Практический backlog: в каком порядке делать

Каждый пункт — отдельный проверяемый результат/небольшой PR. Не ждать завершения глобального рефакторинга, чтобы получить первый эксперимент.

### Итерация 0 — взять уже подтверждённый baseline

- [x] Новый код с готовыми лучшими артефактами получил `.957` на submission — подтверждено пользователем.
- [x] Повторное доказательство совместимости, старый-vs-новый notebook и стартовое переобучение исключены из задач.
- [ ] Сохранить идентичность успешного run/config/bundle. Все его веса и runtime-настройки остаются исходной точкой.
- [ ] Собрать уже имеющиеся predictions/evidence/reports на train; inference выполнить только для недостающих данных.
- [ ] Зафиксировать development/control movie lists и локальный baseline score.

### Итерация 1 — минимальный replay и анализ ошибок, без train

- [ ] Достать нужный prefix из сохранённых графов либо добавить snapshot hooks перед недостающим inference.
- [ ] Тонкий prefix/suffix replay-wrapper; проверить, что именно этот новый wrapper не меняет baseline на том же кэше.
- [ ] Stage deltas и error atlas; existing-candidate oracle для наиболее заметных потерь.
- [ ] Expanded/joint oracle реализовывать по потребности, не ждать полного инструментария до первого эксперимента.
- [ ] Никаких обязательных training bank builders для всех стадий.

### Итерация 2 — первые улучшения с теми же весами

- [ ] N1: CandidateGRAFT threshold sweep и pre/final ablation.
- [ ] N2: cleanup, node budget и smoothing с отчётом TP/FP/FN.
- [ ] N3: knobs текущего Motion и assignment, с последовательным replay истории.
- [ ] N4/N5: division/ownership/EdgeGRAFT guards и DeepCenter/gap thresholds.
- [ ] Полезные изменения проверить совместно и на control; принять только подтверждённые gains в новый baseline config. Все model artifacts пока прежние.

### Итерация 3 — выбрать и обучить один challenger

- [ ] По error budget выбрать одну стадию, измерить candidate/selection ceiling и записать гипотезу. Не переучивать всю цепочку.
- [ ] Подготовить или переиспользовать только её training bank; настроить folds и validation новой модели при фиксированных остальных артефактах.
- [ ] Дешёвый кандидат: CatBoost ownership. Кандидат на основной train: Motion M0/M1 с predicted histories, затем M2/M3/M4 по результатам. Это альтернативы по приоритету ошибок, не обязательный пакет запусков.
- [ ] Сравнить новый artifact с соответствующим лучшим frozen artifact по полному suffix; старый не удалять.
- [ ] После изменения prefix обновить только зависимые replay-кэши/features и банки последующих запланированных экспериментов. Остальные модели не переобучать автоматически; проверить их работу на новом prefix.

### Итерация 4 — coordinated graph repair, если это следующий источник gain

- [ ] EdgeGRAFT E0: готовые ranker/gate + альтернативное разрешение конфликтов; обучение здесь ещё не обязательно.
- [ ] E1: свежий банк и model challenger только при доказанном selection/score bottleneck.
- [ ] E2/E3/E4: utility + совместимые transactions внутри компонентов.
- [ ] Ownership alternatives и guards проверить вместе с новым EdgeGRAFT.

### Итерация 5 — division и missing nodes

- [ ] D1/D2: сильнее source/pair scoring и temporal reranker.
- [ ] Если candidate ceiling подтверждён — D3 missing-daughter/event-time repair.
- [ ] Gap/DeepCenter acceptance model на той же инфраструктуре crop/transaction banks.

### Итерация 6 — глобальная система

- [ ] Общий endpoint proposal/scoring/decoder interface.
- [ ] Graph corruption → pseudo-label/SSL по ablation, не всё одновременно.
- [ ] Complementary ensembles, совместная calibration и бюджет serving.
- [ ] Полная проверка на целевом Kaggle-профиле и финальная упаковка.

Практически первые действия: **взять работающий `.957` и имеющиеся результаты → дополучить только недостающие inference-кэши → локальная оценка и ошибки → N1–N5 с готовыми весами → выбрать один challenger**. Нет стартового этапа «переобучить baseline», «снова доказать совместимость» или «собрать training banks всех стадий».

## 18. Проверки перед принятием модели и перед submission

Это regression-проверки **будущих изменений**, не требование заново проверять уже подтверждённый `.957`. Train-specific пункты относятся только к новым обучаемым моделям; для threshold/logic-only эксперимента применяются graph/replay/selection проверки.

### Минимальный набор regression-тестов новых компонентов

- Train/serving feature parity на одном входном графе, включая history и edge probabilities.
- Отсутствие утечки GT в proposals/features; unknown supervision не становится FP/negative автоматически.
- Outer/inner fold isolation и корректная маршрутизация artifact на видео.
- Empty candidates, одна клетка, отсутствующие кадры/history, NaN/Inf, missing probabilities и валидные пустые массивы.
- Round-trip export/load для каждой модели; feature order/dtypes/schema version проверяются строго.
- Cache invalidation при смене содержимого по тому же пути; failed build не портит живой cache.
- Atomic transactions: нет dangling/duplicate edges, два родителя не присваиваются одной клетке, допустимая cardinality и временной порядок сохранены.
- Replay parity с baseline; full suffix запускается после node/coordinate/ownership changes.
- Скоростные изменения проверены отдельно от modeling изменений.

### Перед promotion в новый baseline

- [ ] Полный набор видео, complete evaluation, finite scores, same scorer/data/evaluation contract.
- [ ] Эксперимент действительно применён: для новой модели — paths/hashes/version, для logic-only — изменённые knobs и неизменные веса. Нет silent fallback к прежнему экспериментальному варианту.
- [ ] Для новой модели OOF оценка не использует final-fit artifact. Пороги/features/blend не выбраны на outer-validation/control; frozen threshold-tuning не переименован в новую OOF-оценку.
- [ ] Есть ΔS и обе компоненты, per-embryo/worst cases, node ratio и runtime/fallback report.
- [ ] Общий gain пересчитан поверх уже принятого baseline; отдельные gains не сложены арифметически.
- [ ] Если менялся общий serving-код, regression-тесты защищают неизменный baseline recipe. Для экспериментального результата проверяется его собственная воспроизводимость, а не равенство старому CSV. До начала изменений повторный compatibility/golden run не требуется.

### Перед Kaggle submission

- [ ] Согласованный immutable bundle: weights, reports, thresholds, feature schemas, fold policy и code versions.
- [ ] Notebook использует нужный config/bundle, а не локальные defaults. Новый профиль проверен целиком.
- [ ] Test movie list получен из фактических test inputs: не полагаться на `panel: smoke` из локального конфига.
- [ ] Полный dry run на целевых ограничениях памяти/времени; H200 server speed не переносится автоматически на Kaggle GPU.
- [ ] Нет неожиданных emergency/fallback, пустых или повторённых datasets; submission schema, IDs и ссылки рёбер валидны.
- [ ] Воспроизводимые checksums, сохранён прошлый bundle для отката. Сам upload выполняется отдельно по решению пользователя.

## Итог

Сильная ставка сейчас — **больше информации о реальном движении и делении + лучшее совместное принятие решений**, а не просто более тяжёлая модель на старом банке.

**Старт — уже работающий `.957` со всеми лучшими артефактами.** Сначала кэши, локальная оценка и улучшение решений без обучения. Затем — точечные challengers по доказанному error budget, при неизменных остальных моделях. Motion с predicted history — сильная train-гипотеза, а не обязательное стартовое переобучение.

Главные направления крупного gain сохраняются: coordinated EdgeGRAFT/ownership и temporal division repair с восстановлением отсутствующих кандидатов. P1/P2/C остаются замороженными; текущие лучшие артефакты остальных стадий заменяются только после подтверждённого выигрыша конкретного challenger.
