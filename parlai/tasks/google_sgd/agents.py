#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import parlai.tasks.google_sgd.build as build_
import parlai.core.tod.tod_core as tod
import parlai.core.tod.tod_agents as tod_agents
from parlai.core.tod.tod_core import SerializationHelpers
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from fuzzywuzzy import fuzz
from parlai.utils.io import PathManager


class GoogleSGDParser(tod_agents.TodStructuredDataParser):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--delex", type="bool", default=False, help="Delexicalize labels"
        )
        parser.add_argument(
            "--filter-dialogue-by-id",
            default="",
            type=str,
            help="Path to a json file of `dialogue_id`s for which we will filter from. Assumes it will contain a map where the keys are a fold and the value is a list of ids",
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        self.fold = self.get_fold(opt)
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "google_sgd")
        if shared is None:
            # full initialize the teacher as this is not a clone
            build_.build(opt)
        super().__init__(opt, shared)

    def get_fold(self, opt):
        return opt["datatype"].split(":")[0]

    def _load_data(self, fold):
        dataset_fold = "dev" if fold == "valid" else fold
        fold_path = os.path.join(self.dpath, dataset_fold)
        schema_file = os.path.join(fold_path, "schema.json")
        with PathManager.open(schema_file, "r") as f:
            schema_lookup = {}
            for schema in json.load(f):
                schema_lookup[schema["service_name"]] = schema

        dialogues = []
        for filename in glob.glob(f"{fold_path}/dialogues*.json"):
            with PathManager.open(filename, "r") as f:
                dialogues += json.load(f)

        filter_path = self.opt.get("filter_dialogue_by_id", "")
        if len(filter_path) > 0:
            filtered = []
            with open(filter_path) as f:
                dialogues_to_get = json.load(f)[fold]
                for dialogue in dialogues:
                    if dialogue["dialogue_id"] in dialogues_to_get:
                        filtered.append(dialogue)
                assert len(filtered) == len(
                    dialogues_to_get
                ), f"Different number of dialogues found than requested. Are you sure you've got the right form of Google SGD? Did you filter for dialogue ids correctly? len(filtered) = {len(filtered)}, len(dialogues_to_get) = {len(dialogues_to_get)}"
            dialogues = filtered
        return schema_lookup, dialogues

    def _get_api_call_and_results(self, sys_turn):
        api_call = {}
        api_resp = {}
        for frame in sys_turn["frames"]:
            if "service_call" in frame:
                # API CALL
                for slot_type, slot_value in frame["service_call"][
                    "parameters"
                ].items():
                    if slot_value:
                        api_call[
                            f"{slot_type.strip()}"
                        ] = SerializationHelpers.inner_list_join(slot_value)
                api_call[tod.STANDARD_API_NAME_SLOT] = frame["service_call"]["method"]
                assert "service_results" in frame

            # API Resp
            if "service_results" in frame:
                api_resp = {}
                service_results = frame["service_results"]
                if len(service_results) > 0:
                    for key, value in service_results[0].items():
                        api_resp[key] = SerializationHelpers.inner_list_join(value)
        return api_call, api_resp

    def _get_apis_in_domain(self, schema, domain):
        """
        Google SGD includes extra information with the call, so remove these.
        """
        result = {}
        for intent in schema[domain].get("intents", {}):
            here = {}
            if "required_slots" in intent and len(intent["required_slots"]) > 0:
                here[tod.STANDARD_REQUIRED_KEY] = intent["required_slots"]
            if "optional_slots" in intent and len(intent["optional_slots"]) > 0:
                here[tod.STANDARD_OPTIONAL_KEY] = intent["optional_slots"]
            if "result_slots" in intent:
                here["results"] = intent["result_slots"]
            result[intent["name"]] = here
        return result

    def _get_intent_groundinging(self, schema, domains):
        """
        Returns map where keys are intents and values are names of required/optional
        slots.

        We do not care about `result_slots` or default values of optional slots.
        """
        result = []
        for domain in domains:
            apis = self._get_apis_in_domain(schema, domain)
            for intent, params in apis.items():
                here = {}
                here[tod.STANDARD_API_NAME_SLOT] = intent
                if tod.STANDARD_REQUIRED_KEY in params:
                    here[tod.STANDARD_REQUIRED_KEY] = params[tod.STANDARD_REQUIRED_KEY]
                if (
                    tod.STANDARD_OPTIONAL_KEY in params
                    and len(params[tod.STANDARD_OPTIONAL_KEY]) > 0
                ):
                    here[tod.STANDARD_OPTIONAL_KEY] = params[
                        tod.STANDARD_OPTIONAL_KEY
                    ].keys()
                result.append(here)
        return result

    def _get_all_service_calls(self, turns):
        """
        Searches through all turns in a dialogue for any service calls, returns these.
        """
        results = []
        for turn in turns:
            for frame in turn["frames"]:
                if "service_call" in frame:
                    call = frame["service_call"]
                    item = call["parameters"]
                    item[tod.STANDARD_API_NAME_SLOT] = call["method"]
                    results.append(item)
        return results

    def setup_episodes(self, fold):
        """
        Parses Google SGD episodes into TodStructuredEpisode.
        """
        schema_lookup, dialogues = self._load_data(fold)
        result = []
        for dialogue in dialogues:
            domains = {s.split("_")[0].strip() for s in dialogue["services"]}
            turns = dialogue["turns"]
            rounds = []
            for turn_id in range(0, len(turns), 2):
                user_turn = turns[turn_id]
                sys_turn = turns[turn_id + 1]
                api_call, api_results = self._get_api_call_and_results(sys_turn)
                r = tod.TodStructuredRound(
                    user_utt=user_turn["utterance"],
                    api_call_machine=api_call,
                    api_resp_machine=api_results,
                    sys_utt=sys_turn["utterance"],
                )
                rounds.append(r)
            # Now that we've got the rounds, make the episode
            episode = tod.TodStructuredEpisode(
                domain=SerializationHelpers.inner_list_join(domains),
                api_schemas_machine=self._get_intent_groundinging(
                    schema_lookup, set(dialogue["services"])
                ),
                goal_calls_machine=self._get_all_service_calls(turns),
                rounds=rounds,
                delex=self.opt.get("delex"),
                extras={"dialogue_id": dialogue["dialogue_id"]},
            )
            result.append(episode)
        # check if the number of episodes should be limited and truncate as required
        return result

    def get_id_task_prefix(self):
        return "GoogleSGD"


class GoogleSGDDSTTeacher(GoogleSGDParser, tod_agents.TodUserSimulatorTeacher):
    """
    This Teacher is responsible for performing the task of Dialogue State Tracking. It
    can be used to evaluate LM on JGA (Joint Goal Accuracy) metric (as shown in.

    [SimpleTOD](https://arxiv.org/abs/2005.00796) and
    [Soloist](https://arxiv.org/abs/2005.05298)).
    """

    SLOT_ENTRY_SEPARATOR = ", "

    def fuzzy_match(self, gt_strings: List[str], predicted_string: str) -> bool:
        fuzzy_match_scores = [
            fuzz.token_sort_ratio(gt_string, predicted_string) / 100.0
            for gt_string in gt_strings
        ]
        return max(fuzzy_match_scores)

    def compare_slot_values(self, gt_slots, predicted_slots, service_slot) -> bool:
        service_slot_name = service_slot["name"]
        if (
            service_slot_name not in gt_slots
            and service_slot_name not in predicted_slots
        ):
            return True
        if service_slot["is_categorical"]:
            return gt_slots.get(service_slot_name, [""])[0] == predicted_slots.get(
                service_slot_name, ""
            )

        else:

            return self.fuzzy_match(
                gt_slots.get(service_slot_name, [""]),
                predicted_slots.get(service_slot_name, ""),
            )

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        """
        Adapted from https://github.com/google-research/google-research/blob/4b5b9ceb480
        f58fddb289618597187325e8540c3/schema_guided_dst/metrics.py#L245 Slot value is
        considered equal if they match exactly for categorical slots or if they fuzzy
        match for non categorical.
        """

        gt_slots = teacher_action["slots"]

        predicted_slots = {}
        if model_response.get("text"):
            for slots in model_response["text"].split(self.SLOT_ENTRY_SEPARATOR):
                if slots:
                    slot_info = slots.split(" ", maxsplit=1)
                    if len(slot_info) == 2:
                        slot_name, slot_val = slot_info
                        predicted_slots[slot_name] = slot_val

        all_slots_correct = True

        for service_slot in teacher_action["service"]["slots"]:
            are_slots_equal = self.compare_slot_values(
                gt_slots, predicted_slots, service_slot
            )
            all_slots_correct = all_slots_correct and are_slots_equal
            if service_slot["name"] in predicted_slots:
                self.metrics.add("slot_p", AverageMetric(are_slots_equal))
            if service_slot["name"] in gt_slots:
                self.metrics.add("slot_r", AverageMetric(are_slots_equal))

        self.metrics.add("jga", AverageMetric(all_slots_correct))

    def setup_data(self, fold):
        schema_lookup, dialogues = self._load_data(fold)
        for dialogue in dialogues:
            turns = dialogue["turns"]
            context = []
            for turn_id in range(0, len(turns), 2):
                frames = turns[turn_id]["frames"]
                user_utterance = turns[turn_id]["utterance"]
                sys_utterance = turns[turn_id + 1]["utterance"]
                context.append(f"<user> {user_utterance} <system> {sys_utterance}")

                for frame in frames:
                    slot_values = frame["state"]["slot_values"]

                    label = self.SLOT_ENTRY_SEPARATOR.join(
                        [
                            f"{slot_name} {slot_val[0]}"
                            for slot_name, slot_val in slot_values.items()
                        ]
                    )

                    yield {
                        "text": " ".join(context),
                        "label": label,
                        "slots": slot_values,
                        "type": "text",
                        "service": schema_lookup[frame["service"]],
                    }, True


class SystemTeacher(GoogleSGDParser, tod_agents.TodSystemTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass


class UserSimulatorTeacher(GoogleSGDParser, tod_agents.TodUserSimulatorTeacher):
    pass


class StandaloneApiTeacher(GoogleSGDParser, tod_agents.TodStandaloneApiTeacher):
    pass


class SingleGoalAgent(GoogleSGDParser, tod_agents.TodSingleGoalAgent):
    pass


class GoalAgent(GoogleSGDParser, tod_agents.TodGoalAgent):
    pass


class ApiSchemaAgent(GoogleSGDParser, tod_agents.TodApiSchemaAgent):
    pass


class UserUttAgent(GoogleSGDParser, tod_agents.TodUserUttAgent):
    pass


class ApiCallAndSysUttAgent(GoogleSGDParser, tod_agents.TodApiCallAndSysUttAgent):
    pass
