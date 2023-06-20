"""This module provides a custom progress bar for PyTorch Lightning, allowing to see the global remaining time."""


from typing import Any, Optional
import lightning.pytorch as pl
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rich.progress import Task, TaskID


class MyProgressBar(RichProgressBar):
    """Custom progress bar for PyTorch Lightning, allowing to see the global remaining time."""

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(description="green_yellow",
                                                           progress_bar="green1",
                                                           progress_bar_finished="green1",
                                                           batch_progress="green_yellow",
                                                           time="grey82",
                                                           processing_speed="grey82"
                                                           ,metrics="grey82",),
        console_kwargs: Optional[dict[str, Any]]= None,
    ) -> None:
        super().__init__(refresh_rate, leave, theme, console_kwargs)

        self.epoch_progress_bar_id: Optional[TaskID] = None

    def _reset_progress_bar_ids(self) -> None:
        super()._reset_progress_bar_ids()
        self.epoch_progress_bar_id = None

    @property
    def epoch_progress_bar(self) -> Task:
        assert self.progress is not None
        assert self.epoch_progress_bar_id is not None
        return self.progress.tasks[self.epoch_progress_bar_id]
    
    @property
    def validation_description(self) -> str:
        return "👌 Validation"
    
    @property
    def test_description(self) -> str:
        return "🧪 Testing"

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)

        total_epochs = trainer.max_epochs if trainer.max_epochs is not None else 0
        epochs_description = "💪 [green_yellow]Training"

        if self.epoch_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.epoch_progress_bar_id is None:
                self.epoch_progress_bar_id = self._add_task(total_epochs, epochs_description)
            else:
                self.progress.reset(
                    self.epoch_progress_bar_id,
                    total=total_epochs,
                    description=epochs_description,
                    visible=True,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._update(  # TODO: if max_epochs == -1
            self.epoch_progress_bar_id,
            self.trainer.global_step / self.trainer.estimated_stepping_batches * self.trainer.max_epochs,  # type: ignore
        )
    
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_disabled:
            return
        total_batches = self.total_train_batches
        train_description = "🦇 [green_yellow]Batching"

        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering with "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_batches, train_description)
            else:
                self.progress.reset(
                    self.train_progress_bar_id, total=total_batches, description=train_description, visible=True
                )

        self.refresh()
