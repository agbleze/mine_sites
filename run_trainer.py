from argparse import ArgumentParser
import os
from model_trainer import trigger_training_process, get_model
from data_obj import get_data, get_tiff_img
import json


def parse_args():
    parser = ArgumentParser(description="Run Model Trainer")
    parser.add_argument('--train_img_dir', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Path to validation images directory')
    parser.add_argument('--train_target_file', type=str, required=True, help='Path to training target file')
    parser.add_argument('--val_target_file', type=str, required=True, help='Path to validation target file')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument("--save_train_results_as", type=str, default="train_results.json", help="File path to save training results json")
    return parser.parse_args()


def main():
    args = parse_args()
    train_img_dir = args.train_img_dir
    val_img_dir = args.val_img_dir
    train_target_file = args.train_target_file
    val_target_file = args.val_target_file
    num_epochs = args.num_epochs
    save_train_results_as = args.save_train_results_as
    
    train_dl, val_dl = get_data(train_img_dir=train_img_dir, 
                            val_image_dir=val_img_dir,
                            target_file_has_header=True, 
                            loader=get_tiff_img,
                            return_all_bands=True, batch_size=10
                            )

    model, loss_fn, optimizer = get_model()

    result = trigger_training_process(train_dataload=train_dl, val_dataload=val_dl,
                                    model=model, loss_fn=loss_fn,
                                    optimizer=optimizer, 
                                    num_epochs=num_epochs,
                                    )
    
    
    with open(save_train_results_as, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()   