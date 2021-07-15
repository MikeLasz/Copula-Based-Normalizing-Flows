from nflows.distributions.base import Distribution
import torch

class ProdDist(Distribution):
    """A Product Distribution with arbitrary marginals."""

    def __init__(self, shape, marginals):
        super().__init__()
        self._shape = torch.Size(shape)
        self.d = self._shape[0]
        self.marginals = marginals

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        log_prob = 0
        for dim in range(self.d):
            log_prob += self.marginals[dim].log_prob(inputs[:, dim])
        return log_prob # i guess i need to transform the tensor inputs to a np array

    def _sample(self, num_samples, context):
        if context is None:
            dim_wise_sample = []
            for dim in range(self.d):
                dim_wise_sample.append(self.marginals[dim].sample([num_samples]))
            return torch.stack(dim_wise_sample, 1)
        else:
            raise NotImplementedError

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)