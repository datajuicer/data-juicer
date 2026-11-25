from data_juicer.ops.base_op import OPERATORS, Pipeline

OP_NAME = "ray_repartition_pipeline"


@OPERATORS.register_module(OP_NAME)
class RayRepartitionPipeline(Pipeline):
    """Repartition a Ray Dataset into a target number of blocks.

    This operator performs dataset-level block repartitioning through Ray
    Dataset's `repartition` API. It is intended for Ray executor pipelines only
    because local datasets do not expose Ray Dataset blocks.
    """

    def __init__(
        self,
        num_blocks: int = 1,
        shuffle: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param num_blocks: target number of Ray Dataset blocks.
        :param shuffle: whether to shuffle records during repartition.
        """
        super().__init__(*args, **kwargs)
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        self.num_blocks = num_blocks
        self.shuffle = bool(shuffle)

    def run(self, dataset, *, exporter=None, tracer=None):
        from data_juicer.core.data import NestedDataset

        if isinstance(dataset, NestedDataset):
            raise RuntimeError(
                "ray_repartition_pipeline requires Ray executor because local datasets do not have blocks"
            )
        return dataset.repartition(num_blocks=self.num_blocks, shuffle=self.shuffle)
