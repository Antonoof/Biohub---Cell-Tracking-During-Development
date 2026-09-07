from biohub.modules.detect import (
    PredictConfig,
    main,
    predict_from_job,
    predict_movies,
    predict_video,
)

__all__ = ['PredictConfig', 'main', 'predict_from_job', 'predict_movies', 'predict_video']

if __name__ == '__main__':
    raise SystemExit(main())
