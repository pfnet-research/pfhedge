import torch


class _TestBSModule:
    def _assert_shape(self, module, method_name):
        method = getattr(module, method_name)

        N = 2
        M_1 = 10
        M_2 = 11
        H_in = len(module.inputs())

        input = torch.empty((N, H_in))
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N,))

        input = torch.empty((N, M_1, H_in))
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N, M_1))

        input = torch.empty((N, M_1, M_2, H_in))
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N, M_1, M_2))

    def assert_shape_delta(self, module):
        self._assert_shape(module, "delta")

    def assert_shape_gamma(self, module):
        self._assert_shape(module, "gamma")

    def assert_shape_price(self, module):
        self._assert_shape(module, "price")

    def assert_shape_forward(self, module):
        N = 2
        M_1 = 10
        M_2 = 11
        H_in = len(module.inputs())

        input = torch.empty((N, H_in))
        out = module(input)
        assert out.size() == torch.Size((N, 1))

        input = torch.empty((N, M_1, H_in))
        out = module(input)
        assert out.size() == torch.Size((N, M_1, 1))

        input = torch.empty((N, M_1, M_2, H_in))
        out = module(input)
        assert out.size() == torch.Size((N, M_1, M_2, 1))
