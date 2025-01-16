import os
import tensorflow as tf # type: ignore
from constants.const import globalVar
import logging
from typing import Optional
from config.cfg import TrainingConfig # type: ignore

from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from .DPFPruning import DPFPruning # type: ignore
import numpy as np # type: ignore

def debug_mode():
    return bool(os.environ.get("UNAS_DEBUG"))

class ModelTrainer:
    """Trains Keras models according to the specified config."""
    def __init__(self, training_config: TrainingConfig):
        self.log = logging.getLogger("Model trainer")
        self.config = training_config
        self.distillation = training_config.distillation
        self.pruning = training_config.pruning
        self.dataset = training_config.dataset

    def train_and_eval(self, model: tf.keras.Model,
                       epochs: Optional[int] = None, sparsity: Optional[float] = None):
        dataset = self.config.dataset
        batch_size = self.config.batch_size
        sparsity = sparsity or 0.0
        if globalVar.appName == 'solar':
          train = dataset.train_dataset()
          val = dataset.validation_dataset()
        elif globalVar.appName == 'speech':
          train = dataset.train_dataset() \
              .shuffle(batch_size * 8) \
              .batch(batch_size) \
              .prefetch(tf.data.experimental.AUTOTUNE)

          val = dataset.validation_dataset() \
              .batch(batch_size) \
              .prefetch(tf.data.experimental.AUTOTUNE)
        elif globalVar.appName == 'cifar10':
          train = dataset.train_dataset() \
              .shuffle(batch_size * 8) \
              .batch(batch_size) \
              .prefetch(tf.data.experimental.AUTOTUNE)

          val = dataset.validation_dataset() \
              .batch(batch_size) \
              .prefetch(tf.data.experimental.AUTOTUNE)
        else:
           assert('no app name')



        # TODO: check if this works, make sure we're excluding the last layer from the student
        if self.pruning and self.distillation:
            raise NotImplementedError()

        if self.distillation:
            teacher = tf.keras.models.load_model(self.distillation.distill_from)
            teacher._name = "teacher_"
            teacher.trainable = False

            t, a = self.distillation.temperature, self.distillation.alpha

            # Assemble a parallel model with the teacher and student
            i = tf.keras.Input(shape=dataset.input_shape)
            cxent = tf.keras.losses.CategoricalCrossentropy()

            stud_logits = model(i)
            tchr_logits = teacher(i)

            o_stud = tf.keras.layers.Softmax()(stud_logits / t)
            o_tchr = tf.keras.layers.Softmax()(tchr_logits / t)
            teaching_loss = (a * t * t) * cxent(o_tchr, o_stud)

            model = tf.keras.Model(inputs=i, outputs=stud_logits)
            model.add_loss(teaching_loss, inputs=True)

        if self.dataset.num_classes == 2:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

        if globalVar.appName == "solar":
          model.compile(optimizer=self.config.optimizer(),
                        loss=loss, metrics=[accuracy])
        elif globalVar.appName == "speech" or globalVar.appName == 'cifar10':
          model.compile(optimizer=self.config.optimizer(), metrics=["accuracy"],loss=SparseCategoricalCrossentropy(from_logits=True))


        # TODO: adjust metrics by class weight?
        class_weight = {k: v for k, v in enumerate(self.dataset.class_weight())} \
            if self.config.use_class_weight else None
        # epochs = epochs or self.config.epochs
        epochs = globalVar.epoch_param
        callbacks = self.config.callbacks()
        check_logs_from_epoch = 0

        pruning_cb = None
        if self.pruning and sparsity > 0.0:
            assert 0.0 < sparsity <= 1.0
            self.log.info(f"Target sparsity: {sparsity:.4f}")
            pruning_cb = DPFPruning(target_sparsity=sparsity, structured=self.pruning.structured,
                                    start_pruning_at_epoch=self.pruning.start_pruning_at_epoch,
                                    finish_pruning_by_epoch=self.pruning.finish_pruning_by_epoch)
            check_logs_from_epoch = self.pruning.finish_pruning_by_epoch
            callbacks.append(pruning_cb)
            
        if globalVar.appName == 'solar':
          log = model.fit(np.array(train[0]), np.array(train[1]),epochs=epochs, validation_data=(np.array(val[0]),np.array(val[1])), # type: ignore
                          verbose=1,
                          callbacks=callbacks, class_weight=class_weight, batch_size=batch_size)
          test = dataset.test_dataset()
          _, test_acc = model.evaluate(test[0],test[1])
          return {
            "val_error": 1.0 - max(log.history["val_accuracy"]),
            "test_error": 1.0 - test_acc,
            "pruned_weights": pruning_cb.weights if pruning_cb else None
            # "energy": energy_tot
          }
        elif globalVar.appName == 'speech' or globalVar.appName == 'cifar10':
          print("train.shape before modelFit: ",dataset.input_shape)
          log = model.fit(train, epochs=epochs, validation_data=val, 
                        verbose=1 if debug_mode() else 2,
                        callbacks=callbacks, class_weight=class_weight)

          test = dataset.test_dataset() \
              .batch(batch_size) \
              .prefetch(tf.data.experimental.AUTOTUNE)
          _, test_acc = model.evaluate(test, verbose=0)
          # later to add condition when to use this funcgtion

        #   quantised_accuracy(model, dataset,1,5,1)

          return {
            "val_error": 1.0 - max(log.history["val_accuracy"]),
            "test_error": 1.0 - test_acc,
            "pruned_weights": pruning_cb.weights if pruning_cb else None
          }


        # print(log.history["val_accuracy"])
        # print(log.history["val_accuracy"][check_logs_from_epoc:])
