import os
import argparse
from ultralytics import YOLO
from datetime import datetime
from functools import partial
from ultralytics.utils import SETTINGS

def train_jde(data_yaml, project_dir, name, epochs=30, batch=32, imgsz=1280, device=0, amp=True, use_mlflow=True):
    # Enable MLflow and/or Comet
    if use_mlflow:
        try:
            import mlflow
            SETTINGS['mlflow'] = True
            print("MLflow logging enabled.")
        except ImportError:
            print("MLflow not found. Install with 'pip install mlflow'")

    # Comet.ml is also supported natively
    try:
        import comet_ml
        SETTINGS['comet'] = True
        print("Comet logging enabled.")
    except ImportError:
        pass

    from tracker.evaluation.mot_callback import mot_eval

    # Initialize model with JDE task
    model = YOLO('yolo11s-jde.yaml', task='jde').load('yolo11s.pt')

    # Add callback for MOT evaluation every N epochs
    model.add_callback("on_val_end", partial(mot_eval, period=max(1, epochs // 5)))

    model.train(
        project='ifbb_jde', # Fixed name for WandB/Loggers
        name=name,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device, 
        imgsz=imgsz,
        amp=amp,            # Use the parameter
        close_mosaic=0,     # Required for JDE
        patience=25,
        tracker='jdetracker.yaml',
        save=True,
        save_json=True,
        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/ifbb_jde/ifbb_jde.yaml")
    parser.add_argument("--project", type=str, required=True, help="GDrive path for results")
    parser.add_argument("--name", type=str, default="ifbb_jde_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow")
    
    args = parser.parse_args()
    
    train_jde(
        data_yaml=args.data,
        project_dir=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=args.amp,
        use_mlflow=not args.no_mlflow
    )
