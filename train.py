import os
import argparse
from ultralytics import YOLO
from datetime import datetime
from functools import partial
from ultralytics.utils import SETTINGS

def train_jde(data_yaml, project_dir, name, epochs=30, batch=32, imgsz=1280, device=0, amp=True, resume=False, use_mlflow=True):
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
    if resume:
        # If resume is a string, use it as the path. Otherwise, use default last.pt
        resume_path = resume if isinstance(resume, str) and os.path.exists(resume) else os.path.join('ifbb_jde', name, 'weights', 'last.pt')
        
        if os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            model = YOLO(resume_path, task='jde')
            resume = True # Set to True for the model.train call
        else:
            print(f"Warning: Resume path {resume_path} not found, starting from scratch.")
            model = YOLO('yolo11s-jde.yaml', task='jde').load('yolo11s.pt')
            resume = False
    else:
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
        amp=amp,            
        resume=resume,      # Pass resume flag
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
    parser.add_argument("--resume", type=str, nargs='?', const='True', default=None, help="Resume training from last.pt or specific path")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow")
    
    args = parser.parse_args()
    
    # Handle resume logic: None -> False, 'True' -> True, 'path/to/weights' -> 'path/to/weights'
    resume_val = args.resume
    if resume_val == 'True':
        resume_val = True
    elif resume_val is None:
        resume_val = False
    
    train_jde(
        data_yaml=args.data,
        project_dir=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=args.amp,
        resume=resume_val,
        use_mlflow=not args.no_mlflow
    )
