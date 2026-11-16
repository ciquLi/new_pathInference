"""Shared final evaluation with streaming native initialization."""
import loader_engine
import evaluate_pipeline


def main():
    loader_engine.configure()
    evaluate_pipeline.fast.enable_sample = loader_engine.enable_sample
    evaluate_pipeline.main()


if __name__ == '__main__':
    main()
