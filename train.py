import os
import argparse
import shutil
from ultralytics import YOLO
from datetime import datetime
from functools import partial
from ultralytics.utils import SETTINGS

def sync_weights(trainer, sync_dir):
    """Callback to sync weights to a persistent directory (e.g. GDrive)."""
    if not sync_dir:
        return
    
    weights_dir = trainer.save_dir / 'weights'
    if weights_dir.exists():
        try:
            target_dir = os.path.join(sync_dir, trainer.args.name, 'weights')
            os.makedirs(target_dir, exist_ok=True)
            for f in os.listdir(weights_dir):
                if f.endswith('.pt'):
                    shutil.copy2(weights_dir / f, os.path.join(target_dir, f))
            # print(f"Synced weights to {target_dir}")
        except Exception as e:
            print(f"Warning: Failed to sync weights to {sync_dir}: {e}")

def train_jde(data_yaml, project_dir, name, epochs=30, batch=32, imgsz=1280, device=0, amp=True, resume=False, use_mlflow=True, patience=25, sync_dir=None):
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
        resume_path = resume if isinstance(resume, str) and os.path.exists(resume) else os.path.join(project_dir, name, 'weights', 'last.pt')
        
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
    
    # Add callback for weight syncing
    if sync_dir:
        print(f"Weight syncing enabled to: {sync_dir}")
        model.add_callback("on_model_save", partial(sync_weights, sync_dir=sync_dir))

    model.train(
        project=project_dir,
        name=name,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device, 
        imgsz=imgsz,
        amp=amp,            
        resume=resume,      # Pass resume flag
        close_mosaic=0,     # Required for JDE
        patience=patience,
        tracker='jdetracker.yaml',
        save=True,
        save_json=True,
        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/ifbb_jde/ifbb_jde.yaml")
    parser.add_argument("--project", type=str, required=True, help="Base directory for results")
    parser.add_argument("--name", type=str, default="ifbb_jde_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--resume", type=str, nargs='?', const='True', default=None, help="Resume training from last.pt or specific path")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow")
    parser.add_argument("--patience", type=int, default=25, help="Epochs to wait for no observable improvement for early stopping")
    parser.add_argument("--sync_dir", type=str, default=None, help="Optional GDrive/Persistent path to sync weights during training")
    
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
        use_mlflow=not args.no_mlflow,
        patience=args.patience,
        sync_dir=args.sync_dir
    )
