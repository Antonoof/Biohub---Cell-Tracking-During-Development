# Detector train: review и ускорения, 2026-09-08

Проверена версия на базе `c514730`, включая новые architectures/losses/augmentations
и `detector_search.py`. Правки в рабочем дереве, без коммита. Это аудит detector
train и его зависимостей, не новое полное ревью всех tracking-стадий.

## Вердикт

До правок запускать 200 trials было рано: найдены ошибки в геометрических labels,
soft matching, AMP, accumulation и Muon для Conv3D. Они исправлены.
Теперь можно переходить к короткому GPU pilot. **Оптимальность скорости и качество
нового обучения на CUDA ещё не подтверждены.** Полные 200 trials следует запускать
после GPU smoke, проверки VRAM/RAM и выбора честного objective.

Нельзя сравнивать новые метрики/траектории обучения со старыми как один и тот же
эксперимент: часть изменений исправляет математическую задачу, а не только runtime.

## Что исправлено

| Проблема | Изменение |
|---|---|
| `rot90` k=1/3 вращал координаты против направления изображения | Координаты согласованы с `torch.rot90`; тесты всех трёх поворотов |
| Translate/scale создавали маски с дырками, а downstream брал первые N узлов | Компактизация узлов и обеих осей GT transition matrix после visibility changes |
| Scale использовал запрошенный factor вместо фактического округлённого размера interpolation | Учтены half-pixel convention, реальные new_h/new_w, crop/pad offsets |
| Time stretch/warp меняли изображения без переноса GT | Эти опции теперь явно запрещены в config-driven detector train до реализации корректных temporal labels |
| Soft coupling имел actual-GT ширину, transition matrix — padded ширину | Матрица обрезается по действительным размерам couplings |
| Rectangular Sinkhorn мог давать row mass > 1 и BCE targets > 1 | После gating строки приведены к sub-probability mass |
| Probability BCE внутри CUDA autocast | Association reductions выполняются в FP32; division auxiliary использует BCEWithLogits |
| Последний неполный accumulation group не делал optimizer step | Финальный group применяется и нормируется на фактическое число microbatches |
| Finite check scaled gradients прерывал штатную адаптацию GradScaler | Unscale/clip/step/update; EMA не обновляется при overflow-skipped step |
| BatchNorm повторно обновлял running statistics при checkpoint recomputation | Во время recomputation running buffers сохраняются и восстанавливаются |
| Muon не flatten-ил Conv3D weights | 5D convolution weights обрабатываются как output × flattened-input matrix |
| Прямой train/CLI не гарантировал seed модели | Seed применяется до создания модели; CLI учитывает deterministic |
| Training peak injection/soft targets попадали в validation | Validation использует hard matching без дополнительных training peaks |
| Search менял GT matching tolerance/algorithm вместе с моделью | Validation фиксирована на greedy / distance 5.0; протокол записан в model config |
| max_frames мог превысить фактическую длину | Ограничение clamped к длине фильма |

## Что ускорено

1. Убран `itertools.cycle(loader)`: он сохранял загруженные батчи целиком и повторял
   сохранённые аугментированные данные. При исчерпании создаётся новый iterator.
2. Dataset хранит ragged metadata; глобально padded targets больше не создаются
   заранее для каждого окна. В конфигурациях включён `batch_padding: true`.
   Границы supervision не меняются; padding ограничен узлами текущего batch.
3. Перекрывающиеся окна одного movie используют общие frame metadata и transition
   tensors. Это кэш внутри загрузки movie, не постоянный кэш между trials.
4. Добавлен LRU нормализованных FP32 кадров **до аугментаций**, ограниченный байтами.
   Сборка окна имеет отдельный storage: аугментации не портят кэш.
5. Hard matched targets строятся gather-индексацией вместо двух one-hot GEMM.
6. Greedy matching векторизован с сохранением nearest-only политики и tie order.
   Hungarian фильтрует выбранные пары пакетно, без чтения каждого CUDA scalar.
7. В training attention выключен расчёт неиспользуемых attention weights,
   используется SDPA path; norm(kv) вычисляется один раз. Eval сохраняет прежний
   need_weights path, чтобы не менять serving numerics этим ускорением.
8. Training pair heads факторизованы. MLP больше не материализует конкатенацию
   query/key для каждой пары; bilinear вычисляется матричными умножениями.
   State-dict keys сохранены. Outputs и gradients проверены с FP32 tolerance.
9. Feature indexing и binary detection targets векторизованы. Проверка collisions
   перенесена из каждого GPU batch в одно предупреждение на movie при загрузке.
10. Gaussian target строится через точное расстояние до ближайшего GT в CPU worker.
    Это тот же максимум изотропных Gaussians, без radius truncation и округления
    координат. Запросы к KDTree идут блоками, target передаётся на GPU. Больше нет
    полного 3D exponential volume отдельно для каждого GT на training GPU.
11. Blur разделён на горизонтальную и вертикальную свёртки с теми же границами.
12. Проверки gradients и перенос counts на CPU сгруппированы; уменьшено количество
    host synchronizations. Добавлен epoch callback для pruning.

### Что реально измерено

CPU, PyTorch 2.7.1, один Torch thread, median из пяти повторов после warmup.
Это синтетические микробенчмарки отдельных операций, **не ускорение train epoch**.
Для pair heads измерен forward с построением autograd graph, без backward.

| Операция / размер | Было, ms | Стало, ms | Отношение |
|---|---:|---:|---:|
| Greedy, 512 detections × 256 GT | 2.313 | 0.094 | 24.6× |
| Bilinear pair head, B=2, 32×256 pairs, H=128 | 4.107 | 0.206 | 20.0× |
| Gaussian target, 128 GT, 16×64×64 | 22.564 | 16.884 | 1.34× |
| Hard targets, 512 GT, 256 detections | 0.318 | 0.280 | 1.13× |
| MLP pair head, B=2, 32×256 pairs, H=128 | 9.524 | 9.654 | 0.99× |
| Blur, W=2, 16×64×64, sigma=1.2 | 0.697 | 0.680 | 1.02× |

Для MLP и blur существенный выигрыш времени на CPU **не доказан**. Мотивировка
MLP — сокращение промежуточного pair tensor/вычислений, но реальный CUDA gain
нужно проверить. Gaussian gain отдельно не включает overhead worker/H2D transfer.

## Настройка первых GPU запусков

Начальная конфигурация для сравнения runtime, не универсальный optimum:

```yaml
batch_padding: true
frame_cache_mb: 256
num_workers: 4
data_parallel: false
gradient_checkpointing: false
```

`frame_cache_mb` — лимит **на dataset на worker**, не общий лимит процесса.
При четырёх train и четырёх validation workers это до 2 GiB frame caches плюс
prefetch, batches и metadata. В основных YAML кэш оставлен выключенным (`0.0`),
чтобы не увеличивать RAM всех старых запусков без измерения. Сравнить 0/256/512
на целевой машине; при случайных окнах и маленьком кэше hit rate может быть низким.

Не включать `gradient_checkpointing` ради скорости: он нужен, когда иначе не
помещается полезный batch. Precision сначала отдельно проверить на целевой GPU;
в search уже задан BF16, но не всякая GPU поддерживает его эффективно/вообще.

Для каждой конфигурации снять startup time, steady-state seconds/epoch,
validation time, peak allocated/reserved VRAM, RSS всех workers и cache hit rate.
Текущий лог `data/forward/backward` основан на host wall-clock и не является
точным CUDA kernel profile из-за асинхронного выполнения.

## Как подключить pruning

`train_from_config(cfg, epoch_callback=callback)` и `train(..., epoch_callback=...)`
поддерживают callback `(epoch, metrics)`. Возврат `False` завершает обучение
штатно; исключение `TrialPruned` закрывает writer и освобождает ссылки на loaders.
Для одного fold внутри objective:

```python
def on_epoch(epoch, metrics):
    trial.report(metrics['competition_metric'], step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()

train_from_config(cfg, epoch_callback=on_epoch)
```

Сам callback не создаёт study/pruner. Для multi-fold runner нужны уникальные
report steps и явный протокол агрегации; нельзя просто повторять epoch=0 на каждом
fold в одном trial. API report/should_prune описан в
[документации Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

## Что остаётся важным перед 200 trials

1. **Objective ещё proxy, не official tracking metric.** `pair_event_counts`
   получает GT matrix только для matched detections: GT-рёбра с отсутствующим
   endpoint исчезают из этой матрицы. Division считается как бинарное событие
   source с несколькими детьми, а не проверка точного daughter set. Окна
   перекрываются, counts повторяются. `pooled_oof_score` складывает именно эти
   proxy counts. Фиксированный matching убирает один перекос, но не делает
   метрику официальной. Обязательна перепроверка shortlist полным serving/evaluate
   на disjoint panel; до этого нельзя обещать, что Optuna улучшает LB.
2. **Не тратить 200 × 5 folds × до 100 epochs на стартовый широкий search.**
   Моя рекомендация: сначала pilot нескольких разных конфигураций, затем pruning
   на фиксированном fold/budget, а полный CV — shortlist. Сначала проверить,
   что ranking ранних epochs действительно связан с поздним качеством.
   Search одновременно включает множество условных параметров, разные epochs и
   несколько всегда ненулевых auxiliary weights: 200 trials легко окажутся
   слишком разреженным поиском. Это вопрос дизайна эксперимента, не throughput.
3. **Shared preprocessing между trials ещё не реализован.** В данных 19 900
   кадров; после downsample=(1,4,4) это примерно 19.43 GiB FP32 изображений без
   metadata. Подготовленный read-only downsampled store на локальном NVMe/shared
   mmap — следующий крупный кандидат для устранения повторного Zarr decode.
   Нельзя делать по полной RAM-копии этого store на каждый worker/trial.
   Нужны version/fingerprint и подтверждение идентичности normalization.
4. **Нужно CUDA profiling.** Остались dense O(Nsource×Ntarget) association,
   Python loops по batch/frame, FP32 validation каждый epoch, geometry channels,
   optional deform/grid_sample и per-parameter custom optimizers. Размер выигрыша
   от torch.compile, channels_last_3d, fused optimizers и переноса аугментаций
   на GPU без профиля неизвестен. Эти переключения не включены вслепую.
5. **SAM/Cellpose ветки не сертифицированы.** Они исключены из текущего search
   (`none/conv`). Cellpose backbone находится внутри стороннего объекта, hardcoded
   projection ожидает 32 канала, используется повтор среднего z-slice; SAM берёт
   средний slice и затем усредняет spatial features. Оба backbone вызываются под
   no_grad даже при freeze=False. Не включать их в sweep без отдельного исправления
   device/checkpoint/feature/freeze contracts и smoke с реальными dependency versions.

## Проверки и ограничения

- Финальный `tests/unit + tests/parity`: **183 passed, 2 skipped, 21 deselected**.
  Два skipped — CUDA AMP tests; deselected — существующие slow tests.
- Отдельное временное окружение Python 3.12 / Torch 2.7.1; пользовательское `.venv`,
  pyproject и uv.lock не изменены.
- Unit/parity и новые regression tests покрывают labels после rotation/visibility,
  soft padded matching, gradients pair heads, BN checkpoint buffers, accumulation,
  scaler overflow/EMA, bounded cache, exact Gaussian targets, Muon Conv3D, callback
  и фиксированный validation protocol.
- Настоящий CPU smoke на `44b6_0113de3b`, первых трёх кадрах: два train batch,
  persistent spawn worker, frame cache, translate/rot90, Gaussian target,
  Sinkhorn soft matching, accumulation tail, validation, checkpoint save/reload.
  Train ~4.8 s, validation ~3.2 s, включая startup; tiny model 6 397 parameters.
  Score=0 у случайной tiny model ожидаем; это не проверка сходимости/качества.
- CUDA tests автоматически пропускаются на этой машине; GPU golden/full train/
  фактическое ускорение эпохи/все 200 trials не запускались.
- Ruff и diff whitespace проверяются. Type check изменённых файлов проходит;
  полный ty во временном окружении сообщает 8 NumPy min/max typing diagnostics
  в не затронутых `data/image.py`, `modules/cardinality/head.py`, `modules/unigraft.py`,
  `tests/unit/test_graph.py`. Полный ty нельзя считать зелёным по этому запуску.

Модельный train-path менялся; старые checkpoint keys сохранены. Full inference
golden после этих правок не запускался: существующие parity tests не заменяют его.
