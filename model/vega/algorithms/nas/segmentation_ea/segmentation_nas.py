# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Search algorithm used to search BiSeNet code. Include random search and mutate search."""
from copy import deepcopy
from .conf import SegmentationConfig
from .segmentation_random import SegmentationRandom
from .segmentation_mutate import SegmentationMutate
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from vega.report import ReportServer


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SegmentationNas(SearchAlgorithm):
    """Search algorithm of SegmentationNas."""

    config = SegmentationConfig()

    def __init__(self, search_space=None):
        """Construct the SegmentationNas class.

        :param search_space: Config of the search space
        """
        super(SegmentationNas, self).__init__(search_space)
        self.search_space = search_space
        self.max_sample_random = self.config.max_sample_random
        self.max_sample_mutate = self.config.max_sample_mutate
        self.sample_count = 0
        self.random = SegmentationRandom()
        self.mutate = SegmentationMutate()

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        """
        return (self.sample_count >= self.max_sample_random + self.max_sample_mutate)

    def search(self):
        """Search code of a model."""
        desc = deepcopy(self.search_space)
        search_desc = self.search_space.custom
        if self.sample_count < self.max_sample_random:
            encoding = self.random.search()
        else:
            records = ReportServer().get_pareto_front_records(['nas'])
            if len(records) == 0:
                encoding = self.random.search()
                print('pareto_front_records is None, do random search')
            else:
                encoding = self.mutate.search()
        search_desc['encoding'] = deepcopy(encoding)
        encoding[0] = self.codec.decode(encoding[0])
        search_desc['config'] = encoding
        self.sample_count += 1
        desc['custom'] = search_desc
        return self.sample_count, desc
