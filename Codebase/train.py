import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

from Parameter import train_argparser
from models.D2E2S_Model import D2E2SModel
from models.General import set_seed
from trainer import util, sampling
from trainer.baseTrainer import BaseTrainer
from trainer.entities import Dataset
from trainer.evaluator import Evaluator
from trainer.input_reader import JsonInputReader
from trainer.loss import D2E2SLoss
import warnings

warnings.filterwarnings("ignore")


class D2E2S_Trainer(BaseTrainer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._tokenizer = AutoTokenizer.from_pretrained(args.pretrained_deberta_name)
        self._predictions_path = os.path.join(
            self._log_path_predict, "predicted_%s_epoch_%s.json"
        )
        self._examples_path = os.path.join(
            self._log_path_predict, "sample_%s_%s_epoch_%s.html"
        )
        os.makedirs(self._log_path_result)
        os.makedirs(self._log_path_predict)
        # Keep initial dev-best low so first dev eval is always logged.
        self.max_pair_f1 = -1.0
        self.best_dev_epoch = 0
        self.best_dev_metric = 0.0
        self.result_path = os.path.join(
            self._log_path_result, "result{}.txt".format(self.args.max_span_size)
        )

    def _print_device_info(self, model: torch.nn.Module, stage: str):
        model_device = next(model.parameters()).device
        print(f"[{stage}] args.device: {self.args.device}")
        print(f"[{stage}] model device: {model_device}")
        print(f"[{stage}] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[{stage}] CUDA device count: {torch.cuda.device_count()}")
            print(f"[{stage}] GPU name: {torch.cuda.get_device_name(model_device)}")

    def _preprocess(self, args, input_reader_cls, types_path, train_path, dev_path, test_path):

        train_label, dev_label, test_label = "train", "dev", "test"
        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(dev_label)
        self._init_eval_logging(test_label)

        # loading data
        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            args.neg_entity_count,
            args.neg_triple_count,
            args.max_span_size,
        )
        input_reader.read({train_label: train_path, dev_label: dev_path, test_label: test_path})
        train_dataset = input_reader.get_dataset(train_label)

        # preprocess
        train_sample_count = train_dataset.sentence_count
        updates_epoch = train_sample_count // args.batch_size
        updates_total = updates_epoch * args.epochs

        print("   ", self.args.dataset, "  ", self.args.max_span_size)
        return input_reader, updates_total, updates_epoch

    def _train(
        self, train_path: str, dev_path: str, test_path: str, types_path: str, input_reader_cls
    ):
        args = self.args

        # set seed
        set_seed(args.seed)

        train_label, dev_label, test_label = "train", "dev", "test"
        input_reader, updates_total, updates_epoch = self._preprocess(
            args, input_reader_cls, types_path, train_path, dev_path, test_path
        )
        train_dataset = input_reader.get_dataset(train_label)
        dev_dataset = input_reader.get_dataset(dev_label)
        test_dataset = input_reader.get_dataset(test_label)

        # load model
        config = AutoConfig.from_pretrained(args.pretrained_deberta_name)

        model = D2E2SModel.from_pretrained(
            self.args.pretrained_deberta_name,
            config=config,
            cls_token=self._tokenizer.cls_token_id,
            sentiment_types=input_reader.sentiment_type_count - 1,
            entity_types=input_reader.entity_type_count,
            args=args,
        )
        model.to(args.device)
        self._print_device_info(model, "train")
        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        # create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup * updates_total,
            num_training_steps=updates_total,
        )

        # create loss function
        entity_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        senti_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        sentence_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        compute_loss = D2E2SLoss(
            senti_criterion,
            entity_criterion,
            sentence_criterion,
            model,
            optimizer,
            scheduler,
            args.max_grad_norm,
            args.sentence_loss_weight,
        )
        # eval validation set
        if args.init_eval:
            self._eval(model, dev_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self.train_epoch(
                model, compute_loss, optimizer, train_dataset, updates_epoch, epoch
            )

            # eval on dev set and save best model
            dev_ner_eval, dev_senti_eval, dev_senti_nec_eval, dev_extra_eval = self._eval(
                model, dev_dataset, input_reader, epoch + 1, updates_epoch, label_to_log="dev"
            )
            dev_f1 = float(dev_senti_nec_eval[2])
            is_best = dev_f1 > self._best_results.get("dev", 0)
            
            # Save best model based on dev F1 score
            self._save_best(
                model, 
                self._tokenizer, 
                optimizer, 
                dev_f1, 
                epoch + 1,
                label="dev",
                extra=None
            )
            if is_best:
                self.best_dev_epoch = epoch + 1
                self.best_dev_metric = dev_f1
                print(f"New best dev checkpoint at epoch {self.best_dev_epoch} (metric={self.best_dev_metric:.2f})")
        
        # Load best model and evaluate on test set
        print(f"\nLoading best model with dev set")
        best_model_path = os.path.join(self._save_path, 'model_dev_best')
        if os.path.exists(best_model_path):
            model = D2E2SModel.from_pretrained(
                best_model_path,
                config=model.config,
                cls_token=self._tokenizer.cls_token_id,
                sentiment_types=input_reader.sentiment_type_count - 1,
                entity_types=input_reader.entity_type_count,
                args=args,
            )
            model.to(args.device)
            self._print_device_info(model, "eval-best")
            print("\n" + "="*80)
            print(f"FINAL TEST RESULTS (Best Model from Dev Set, epoch={self.best_dev_epoch}, metric={self.best_dev_metric:.2f})")
            print("="*80)
            self._eval(model, test_dataset, input_reader, self.best_dev_epoch, updates_epoch, label_to_log="test_final")
            print("="*80)
        else:
            print(f"Best model path not found: {best_model_path}")

    def train_epoch(
        self,
        model: torch.nn.Module,
        compute_loss: D2E2SLoss,
        optimizer: Optimizer,
        dataset: Dataset,
        updates_epoch: int,
        epoch: int,
    ):

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        model.zero_grad()

        iteration = 0
        total = dataset.sentence_count // self.args.batch_size
        for batch in tqdm(data_loader, total=total, desc="Train epoch %s" % epoch):
            model.train()
            batch = util.to_device(batch, self.args.device)

            # forward step
            entity_logits, senti_logits, sentence_logits, batch_loss = model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                sentiments=batch["rels"],
                senti_masks=batch["senti_masks"],
                adj=batch["adj"],
            )

            # compute loss and optimize parameters
            epoch_loss = compute_loss.compute(
                entity_logits=entity_logits,
                senti_logits=senti_logits,
                sentence_logits=sentence_logits,
                batch_loss=batch_loss,
                senti_types=batch["senti_types"],
                sentence_types=batch["sentence_types"],
                entity_types=batch["entity_types"],
                entity_sample_masks=batch["entity_sample_masks"],
                senti_sample_masks=batch["senti_sample_masks"],
            )

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(
                    optimizer,
                    epoch_loss,
                    epoch,
                    iteration,
                    global_iteration,
                    dataset.label,
                )

        return iteration

    def _log_train(
        self,
        optimizer: Optimizer,
        loss: float,
        epoch: int,
        iteration: int,
        global_iteration: int,
        label: str,
    ):
        # average loss
        avg_loss = loss / self.args.batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # to_csv
        # log to csv
        self._log_csv(label, "loss", loss, epoch, iteration, global_iteration)
        self._log_csv(label, "loss_avg", avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, "lr", lr, epoch, iteration, global_iteration)

        # log to tensorboard
        self._log_tensorboard(label, "loss", loss, global_iteration)
        self._log_tensorboard(label, "loss_avg", avg_loss, global_iteration)
        self._log_tensorboard(label, "lr", lr, global_iteration)

    def _eval(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        input_reader: JsonInputReader,
        epoch: int = 0,
        updates_epoch: int = 0,
        iteration: int = 0,
        label_to_log: str = None,
    ):

        # Use dataset label if label_to_log not specified
        if label_to_log is None:
            label_to_log = dataset.label

        # Normalize logging label to one of the initialized labels (train/dev/test)
        if label_to_log in self._log_paths:
            log_label = label_to_log
        else:
            base = label_to_log.split('_')[0] if isinstance(label_to_log, str) else dataset.label
            log_label = base if base in self._log_paths else dataset.label

        # create evaluator (store predictions/examples under the dataset's base label)
        evaluator = Evaluator(
            dataset,
            input_reader,
            self._tokenizer,
            self.args.sen_filter_threshold,
            self.args.sentence_filter_threshold,
            self.args.eval_match_mode,
            self._predictions_path,
            self._examples_path,
            self.args.example_count,
            epoch,
            dataset.label,
        )
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        with torch.no_grad():
            model.eval()
            # iterate batches
            total = math.ceil(dataset.sentence_count / self.args.batch_size)
            for batch in tqdm(
                data_loader, total=total, desc="Evaluate epoch %s" % epoch
            ):
                # move batch to selected device
                batch = util.to_device(batch, self.args.device)

                # run model (forward pass)
                result = model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    evaluate=True,
                    adj=batch["adj"],
                )
                entity_clf, senti_clf, rels, sentence_clf = result
                # evaluate batch, entity:tensor(16, 188, 3), senti_clf:tensor(16, 2, 4), rels:tensor(16, 2, 2)
                evaluator.eval_batch(entity_clf, senti_clf, rels, batch, sentence_clf)
            global_iteration = epoch * updates_epoch + iteration
            print_examples = 5 if label_to_log == "test_final" else 0
            print_extra_metrics = label_to_log in {"dev", "test_final"}
            ner_eval, senti_eval, senti_nec_eval, extra_eval = evaluator.compute_scores(
                print_examples=print_examples,
                print_extra_metrics=print_extra_metrics,
            )
            # print(self.result_path)
            self._log_filter_file(ner_eval, senti_eval, senti_nec_eval, extra_eval, evaluator, epoch, label_to_log)
        self._log_eval(
            *ner_eval,
            *senti_eval,
            *senti_nec_eval,
            epoch,
            iteration,
            global_iteration,
            log_label
        )
        return ner_eval, senti_eval, senti_nec_eval, extra_eval

    def _log_filter_file(self, ner_eval, senti_eval, senti_nec_eval, extra_eval, evaluator, epoch, label_to_log="test"):
        # quintuple metric = exact quintuple F1
        f1 = float(senti_nec_eval[2])
        columns = [
            "mic_precision",
            "mic_recall",
            "mic_f1_score",
            "mac_precision",
            "mac_recall",
            "mac_f1_score",
        ]

        def metric_dict(values):
            return {columns[idx]: values[idx] for idx in range(len(columns))}

        # Track and store results for dev and test sets
        if label_to_log == "dev":
            # Only update best when dev improves
            if self.max_pair_f1 < f1:
                ner_dic = metric_dict(ner_eval)
                senti_dic = metric_dict(senti_eval)
                senti_nec_dic = metric_dict(senti_nec_eval)
                self.max_pair_f1 = f1
                with open(self.result_path, mode="a", encoding="utf-8") as f:
                    w_str = "No. {} ： (dev) ....\n".format(epoch)
                    f.write(w_str)
                    f.write("ner_entity: \n")
                    f.write(str(ner_dic))
                    f.write("\n exact_quintuple: \n")
                    f.write(str(senti_nec_dic))
                    if extra_eval:
                        if 'coqe_metrics_index' in extra_eval:
                            f.write("\n coqe_metrics_full_index_match: \n")
                            f.write(str(extra_eval['coqe_metrics_index']))
                        if 'coqe_metrics_span' in extra_eval:
                            f.write("\n coqe_metrics_full_span_match: \n")
                            f.write(str(extra_eval['coqe_metrics_span']))
                        if 'sentence' in extra_eval:
                            f.write("\n sentence_comparative: \n")
                            f.write(str(extra_eval['sentence']))
                        if 'label' in extra_eval:
                            f.write("\n label_only: \n")
                            f.write(str(metric_dict(extra_eval['label'])))
                        if 'exact_quadruple' in extra_eval:
                            f.write("\n exact_quadruple: \n")
                            f.write(str(metric_dict(extra_eval['exact_quadruple'])))
                    f.write("\n")
        elif label_to_log == "test_final":
            ner_dic = metric_dict(ner_eval)
            senti_dic = metric_dict(senti_eval)
            senti_nec_dic = metric_dict(senti_nec_eval)
            with open(self.result_path, mode="a", encoding="utf-8") as f:
                w_str = "No. {} ： (test_final) ....\n".format(epoch)
                f.write(w_str)
                f.write("ner_entity: \n")
                f.write(str(ner_dic))
                f.write("\n exact_quintuple: \n")
                f.write(str(senti_nec_dic))
                if extra_eval:
                    if 'coqe_metrics_index' in extra_eval:
                        f.write("\n coqe_metrics_full_index_match: \n")
                        f.write(str(extra_eval['coqe_metrics_index']))
                    if 'coqe_metrics_span' in extra_eval:
                        f.write("\n coqe_metrics_full_span_match: \n")
                        f.write(str(extra_eval['coqe_metrics_span']))
                    if 'label' in extra_eval:
                        f.write("\n label_only: \n")
                        f.write(str(metric_dict(extra_eval['label'])))
                    if 'sentence' in extra_eval:
                        f.write("\n sentence_comparative: \n")
                        f.write(str(extra_eval['sentence']))
                    if 'exact_quadruple' in extra_eval:
                        f.write("\n exact_quadruple: \n")
                        f.write(str(metric_dict(extra_eval['exact_quadruple'])))
                f.write("\n")
            if self.args.store_predictions:
                evaluator.store_predictions()
                evaluator.store_gold_pred_log()
            if self.args.store_examples:
                evaluator.store_examples()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_params


if __name__ == "__main__":
    arg_parser = train_argparser()
    trainer = D2E2S_Trainer(arg_parser)
    trainer._train(
        train_path=arg_parser.dataset_file["train"],
        dev_path=arg_parser.dataset_file["dev"],
        test_path=arg_parser.dataset_file["test"],
        types_path=arg_parser.dataset_file["types_path"],
        input_reader_cls=JsonInputReader,
    )