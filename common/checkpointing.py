"""
I rolled my own checkpointing for no good reason. Oops.
"""
import os


class Checkpointer:
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.hist = save_folder / "hist.txt"
        self.hist.touch()

    def get_best_model_info(self):
        """
        Returns info on the best model so far.
        """
        files = self.hist.read_text().split("\n")
        actual = [f for f in files if f.endswith(".pt")]
        best_fname = None
        best_loss = float("inf")
        best_epoch = None
        for fname in actual:
            if "batch" in fname:
                continue
            f = fname[:-3]  # remove .pt
            _, epoch_str, loss_str = f.split("_")
            _, epoch = epoch_str.split("=")
            _, loss = loss_str.split("=")
            epoch = int(epoch)
            loss = float(loss)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_fname = fname
        if best_fname is None:
            return None, None, None
        return FOLDER / best_fname, best_epoch, best_loss

    def save_checkpoint(self, model, optimizer, epoch, avg_loss):
        """
        Save the model and optimizer state if the loss has decreased.
        """
        # see if we should save weights
        best_fpath, best_epoch, best_loss = self.get_best_model_info()
        if best_fpath is None or avg_loss < best_loss:
            print(f"Loss decreased from {best_loss} to {avg_loss}, saving...")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            avg_loss = round(avg_loss, 4)
            torch.save(
                checkpoint, FOLDER / f"checkpoint_epochs={epoch}_loss={avg_loss}.pt"
            )
            if best_fpath is not None:
                print("Deleting old best:", best_fpath)
                os.remove(best_fpath)

        # save to history file regardless
        with self.hist.open("a") as f:
            f.write(f"checkpoint_epochs={epoch}_loss={avg_loss}.pt\n")

    def load_checkpoint(self):
        """
        Loads the best model and optimizer states.
        """
        best_fpath, best_epoch, best_loss = self.get_best_model_info()
        if best_fpath is None:
            return None, None
        checkpoint = torch.load(best_fpath)
        return checkpoint, best_epoch
