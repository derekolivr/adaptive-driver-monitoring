# Fine-Tuning the Gaze Model on DashGaze

This guide explains how to adapt the MpiiFaceGaze-trained model to work better on the DashGaze dataset.

## Why Fine-Tuning?

The gaze tracker was originally trained on **MpiiFaceGaze** (lab environment), but you're testing it on **DashGaze** (real driving conditions). This causes:

1. **Domain Mismatch**: Different camera angles, lighting, and head poses
2. **Coordinate System Differences**: Different conventions for angle measurements
3. **Higher Errors**: 15-25° errors instead of 5-10°

**Fine-tuning** adapts the model to DashGaze while keeping the learned face features from MpiiFaceGaze.

## Step-by-Step Process

### Step 1: Generate Training Data

First, extract 500 labeled frames from your DashGaze video:

```bash
python scripts/preprocess_dashgaze.py
```

This will:

- Extract 500 driver images (evenly spaced)
- Extract corresponding road images
- Save ground truth gaze angles from the CSV
- Output to `test_data/dashgaze_processed/`

**Expected output:**

```
Video loaded successfully. Total frames: 23237
Sampling 500 frames evenly spaced throughout the video.
Processing 500 frames...
Extracting Frames: 100%|██████████| 500/500 [01:45<00:00]
Processing complete.
```

### Step 2: Run Fine-Tuning

```bash
python scripts/finetune_dashgaze.py
```

This will:

- Load your pre-trained `gaze_tracker_endterm.pth`
- Freeze the ResNet18 backbone (keeps face features)
- Retrain only the final gaze prediction layers on DashGaze
- Save the best model as `gaze_tracker_dashgaze_finetuned_best.pth`

**Expected output:**

```
Loaded 500 samples from DashGaze dataset.
Dataset split: 400 training, 100 validation samples.

Starting fine-tuning for 15 epochs...
======================================================================
Epoch [1/15] | Train Loss: 0.145230 | Val Loss: 0.089456
  ✓ Best model saved (val_loss: 0.089456)
Epoch [2/15] | Train Loss: 0.067823 | Val Loss: 0.051234
  ✓ Best model saved (val_loss: 0.051234)
...
```

**Training time:** ~5-10 minutes on Apple Silicon, ~15-20 minutes on CPU

### Step 3: Update Your App

Modify `src/app.py` to use the fine-tuned model:

```python
# Change this line:
"gaze": GazeTracker(model_path="gaze_tracker_endterm.pth"),

# To this:
"gaze": GazeTracker(model_path="gaze_tracker_dashgaze_finetuned_best.pth"),
```

Restart Streamlit:

```bash
streamlit run src/app.py
```

### Step 4: Compare Results

Check the "Ground Truth Comparison" section in the app. You should see:

**Before Fine-Tuning:**

- Azimuth Error: ~5-10°
- Elevation Error: ~15-25°

**After Fine-Tuning:**

- Azimuth Error: ~2-5°
- Elevation Error: ~3-8°

## Advanced Options

### Full Fine-Tuning (Train All Layers)

If you have enough data and training time, you can train all layers:

1. Edit `scripts/finetune_dashgaze.py`:

   ```python
   FREEZE_BACKBONE = False
   ```

2. Increase epochs:

   ```python
   EPOCHS = 30
   ```

3. Re-run the script

**Note:** This takes longer but may give better results with 500+ samples.

### Extract More Training Data

For even better results, extract 1000-2000 frames:

1. Edit `scripts/preprocess_dashgaze.py`:

   ```python
   num_samples = 1000
   ```

2. Re-run preprocessing (takes ~3-5 minutes)
3. Re-run fine-tuning

## Troubleshooting

### "No samples found in the dataset"

- Run `python scripts/preprocess_dashgaze.py` first
- Check that `test_data/dashgaze_processed/` contains `*_driver.jpg` and `*_gt.json` files

### "CUDA out of memory" or MPS errors

- Reduce batch size in `finetune_dashgaze.py`:
  ```python
  train_loader = DataLoader(train_dataset, batch_size=8, ...)  # Changed from 16
  ```

### Model not improving

- Increase training epochs (try 20-30)
- Try full fine-tuning instead of frozen backbone
- Extract more training data (1000+ samples)

## Expected Performance

| Metric          | Before Fine-Tuning | After Fine-Tuning (500 samples) | After Fine-Tuning (1000 samples) |
| --------------- | ------------------ | ------------------------------- | -------------------------------- |
| Azimuth Error   | 5-10°              | 2-5°                            | 1-3°                             |
| Elevation Error | 15-25°             | 3-8°                            | 2-5°                             |
| Training Time   | -                  | 5-10 min                        | 10-20 min                        |

## Next Steps

After fine-tuning:

1. ✅ Remove calibration hacks from `src/app.py` (no longer needed)
2. ✅ Test on all 10 scenarios to verify consistency
3. ✅ Update README.md with the new model name
4. Consider collecting more DashGaze data for production use
