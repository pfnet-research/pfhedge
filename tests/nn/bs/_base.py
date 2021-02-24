import torch


class _TestBSModule:
    def _assert_shape(self, module, method_name):
        method = getattr(module, method_name)

        N = 2
        M_1 = 10
        M_2 = 11
        H_in = len(module.features())

        x = torch.empty((N, H_in))
        out = method(*(x[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N,))

        x = torch.empty((N, M_1, H_in))
        out = method(*(x[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N, M_1))

        x = torch.empty((N, M_1, M_2, H_in))
        out = method(*(x[..., i] for i in range(H_in)))
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
        H_in = len(module.features())

        x = torch.empty((N, H_in))
        out = module(x)
        assert out.size() == torch.Size((N, 1))

        x = torch.empty((N, M_1, H_in))
        out = module(x)
        assert out.size() == torch.Size((N, M_1, 1))

        x = torch.empty((N, M_1, M_2, H_in))
        out = module(x)
        assert out.size() == torch.Size((N, M_1, M_2, 1))
