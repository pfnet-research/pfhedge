import torch


class _TestBSModule:
    def _assert_shape(self, module, method_name, device: str = "cpu"):
        method = getattr(module, method_name)

        N = 2
        M_1 = 10
        M_2 = 11
        H_in = len(module.inputs())

        input = torch.zeros((N, H_in)).to(device)
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N,))

        input = torch.zeros((N, M_1, H_in)).to(device)
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N, M_1))

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        out = method(*(input[..., i] for i in range(H_in)))
        assert out.size() == torch.Size((N, M_1, M_2))

    def assert_shape_delta(self, module, device: str = "cpu"):
        self._assert_shape(module, "delta", device=device)

    def assert_shape_gamma(self, module, device: str = "cpu"):
        self._assert_shape(module, "gamma", device=device)

    def assert_shape_vega(self, module, device: str = "cpu"):
        self._assert_shape(module, "vega", device=device)

    def assert_shape_theta(self, module, device: str = "cpu"):
        self._assert_shape(module, "theta", device=device)

    def assert_shape_price(self, module, device: str = "cpu"):
        self._assert_shape(module, "price", device=device)

    def assert_shape_forward(self, module, device: str = "cpu"):
        N = 2
        M_1 = 10
        M_2 = 11
        H_in = len(module.inputs())

        input = torch.zeros((N, H_in)).to(device)
        out = module(input)
        assert out.size() == torch.Size((N, 1))

        input = torch.zeros((N, M_1, H_in)).to(device)
        out = module(input)
        assert out.size() == torch.Size((N, M_1, 1))

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        out = module(input)
        assert out.size() == torch.Size((N, M_1, M_2, 1))
