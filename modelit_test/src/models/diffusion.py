#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

class GaussianDiffusion:
    """
    Класс для диффузионной модели Гаусса
    Адаптирован из shape-e с улучшенной читаемостью и документацией
    """
    
    def __init__(
        self,
        betas: np.ndarray,
        model_mean_type: str = "epsilon",
        model_var_type: str = "fixed_small",
        loss_type: str = "mse"
    ):
        """
        Инициализация диффузионной модели
        
        Args:
            betas: расписание шума (noise schedule)
            model_mean_type: тип прогнозирования среднего ("epsilon" или "direct")
            model_var_type: тип дисперсии шума ("fixed_small", "fixed_large", или "learned")
            loss_type: тип функции потерь ("mse" или "kl")
        """
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # Преобразуем бета в numpy массив
        self.betas = np.array(betas, dtype=np.float64)
        self.num_timesteps = len(betas)
        
        # Кэшируем различные величины
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Расчеты для диффузии
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Расчеты для posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GaussianDiffusion":
        """
        Создает экземпляр класса из конфигурации
        
        Args:
            config: словарь с параметрами
            
        Returns:
            GaussianDiffusion: инициализированный объект класса
        """
        return cls(
            betas=get_beta_schedule(**config.get("diffusion", {})),
            model_mean_type=config.get("model_mean_type", "epsilon"),
            model_var_type=config.get("model_var_type", "fixed_small"),
            loss_type=config.get("loss_type", "mse")
        )
    
    @classmethod
    def from_pretrained(cls, name: str = "diffusion") -> "GaussianDiffusion":
        """
        Загружает предобученный диффузионный процесс
        
        Args:
            name: имя предобученной конфигурации
            
        Returns:
            GaussianDiffusion: инициализированный объект класса
        """
        # Это заглушка, в реальной имплементации здесь будет загрузка конфигурации
        # от shape-e из предобученной модели
        return cls(
            betas=get_beta_schedule(schedule="linear", num_timesteps=1000),
            model_mean_type="epsilon",
            model_var_type="fixed_small",
            loss_type="mse"
        )
    
    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Получить среднее и дисперсию q(x_t | x_0)
        
        Args:
            x_start: начальное состояние x_0
            t: временные шаги
            
        Returns:
            Кортеж (среднее, дисперсия, логарифм дисперсии)
        """
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)
        return mean, variance, log_variance
    
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Выборка из q(x_t | x_0) с использованием процесса диффузии
        
        Args:
            x_start: начальное состояние x_0
            t: временные шаги
            noise: опционально заданный шум (если None, то генерируется стандартный Гауссов шум)
            
        Returns:
            x_t: состояние в момент времени t
        """
        # Убедимся, что все тензоры находятся на одном устройстве
        device = x_start.device
        t = t.to(device)
        
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        else:
            # Убедимся, что шум находится на том же устройстве
            noise = noise.to(device)
        
        # Формула для выборки x_t = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * eps
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        std = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)
        
        # Убедимся, что mean и std находятся на том же устройстве
        mean = mean.to(device)
        std = std.to(device)
        
        return mean + std * noise
    
    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычислить среднее и дисперсию posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: начальное состояние x_0
            x_t: состояние в момент t
            t: временные шаги
            
        Returns:
            Кортеж (среднее, дисперсия, логарифм дисперсии)
        """
        posterior_mean = (
            extract_and_expand(self.posterior_mean_coef1, t, x_start) * x_start
            + extract_and_expand(self.posterior_mean_coef2, t, x_start) * x_t
        )
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_start)
        posterior_log_variance = extract_and_expand(self.posterior_log_variance_clipped, t, x_start)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Применяет модель для получения параметров distribution p(x_{t-1} | x_t)
        
        Args:
            model: модель, которая предсказывает шум
            x: вход x_t
            t: временные шаги
            model_kwargs: дополнительные аргументы для модели (например, текстовые условия)
            
        Returns:
            Кортеж (среднее, дисперсия, логарифм дисперсии)
        """
        model_output = model(x, t, **model_kwargs)
        
        if self.model_var_type == "learned":
            # Модель предсказывает и среднее, и дисперсию
            model_output, model_log_variance = torch.split(model_output, x.shape[1], dim=1)
            model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in ["fixed_small", "fixed_large"]:
            # Фиксированная дисперсия в зависимости от расписания шума
            model_variance = extract_and_expand(
                self.posterior_variance if self.model_var_type == "fixed_small" else self.betas,
                t, x
            )
            model_log_variance = extract_and_expand(
                self.posterior_log_variance_clipped if self.model_var_type == "fixed_small" else np.log(self.betas),
                t, x
            )
        else:
            raise ValueError(f"Неизвестный model_var_type: {self.model_var_type}")
        
        if self.model_mean_type == "epsilon":
            # Предсказание шума (epsilon)
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            model_mean = self.q_posterior_mean_variance(pred_xstart, x, t)[0]
        elif self.model_mean_type == "direct":
            # Прямое предсказание x_0
            pred_xstart = model_output
            model_mean = self.q_posterior_mean_variance(pred_xstart, x, t)[0]
        else:
            raise ValueError(f"Неизвестный model_mean_type: {self.model_mean_type}")
        
        return model_mean, model_variance, model_log_variance
    
    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Предсказание x_0 на основе предсказанного шума и текущего состояния
        
        Args:
            x_t: текущее состояние
            t: временные шаги
            eps: предсказанный шум
            
        Returns:
            x_0: предсказанное начальное состояние
        """
        sqrt_recip_alphas_cumprod = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        sqrt_recipm1_alphas_cumprod = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * eps
    
    def p_sample(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any] = {}
    ) -> torch.Tensor:
        """
        Сэмплирование из p(x_{t-1} | x_t) с помощью модели
        
        Args:
            model: модель, которая предсказывает шум
            x: вход x_t
            t: временные шаги
            model_kwargs: дополнительные аргументы для модели
            
        Returns:
            x_{t-1}: сэмплированное состояние
        """
        # Убедимся, что все тензоры находятся на одном устройстве
        device = x.device
        t = t.to(device)
        
        # Обработаем model_kwargs, чтобы все тензоры были на одном устройстве
        processed_kwargs = {}
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                processed_kwargs[k] = v.to(device)
            elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                processed_kwargs[k] = v
            else:
                processed_kwargs[k] = v
        
        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t, processed_kwargs)
        noise = torch.randn_like(x)
        
        # Если t=0, не добавляем шум
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        # Формула: x_{t-1} = model_mean + exp(0.5 * model_log_variance) * noise
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample
    
    def p_sample_loop(
        self, 
        model: torch.nn.Module, 
        shape: Tuple[int, ...], 
        model_kwargs: Dict[str, Any] = {},
        device: torch.device = torch.device("cpu"),
        progress: bool = False
    ) -> torch.Tensor:
        """
        Генерация сэмплов с помощью процесса обратной диффузии
        
        Args:
            model: модель, которая предсказывает шум
            shape: форма выходного тензора
            model_kwargs: дополнительные аргументы для модели
            device: устройство для вычислений
            progress: показывать ли прогресс-бар
            
        Returns:
            x_0: сгенерированный сэмпл
        """
        # Начинаем с чистого шума
        x = torch.randn(*shape, device=device)
        
        # Итерация от T до 0
        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                x = self.p_sample(model, x, t, model_kwargs)
        
        return x
    
    def training_losses(
        self, model: torch.nn.Module, x_start: torch.Tensor, t: torch.Tensor, model_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисление функции потерь для обучения
        
        Args:
            model: модель, которая предсказывает шум
            x_start: начальное состояние x_0
            t: временные шаги
            model_kwargs: дополнительные аргументы для модели
            
        Returns:
            Словарь с потерями и промежуточными значениями
        """
        # Получаем устройство от модели и убедимся, что все тензоры на нем
        device = next(model.parameters()).device
        x_start = x_start.to(device)
        t = t.to(device)
        
        # Обработаем model_kwargs, чтобы все тензоры были на одном устройстве
        processed_kwargs = {}
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                processed_kwargs[k] = v.to(device)
            elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                # Для списков строк (например, текстовые описания)
                processed_kwargs[k] = v
            else:
                processed_kwargs[k] = v
        
        # Генерируем шум на том же устройстве
        noise = torch.randn_like(x_start, device=device)
        
        # Получаем x_t с помощью процесса диффузии
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Убедимся, что x_t на правильном устройстве перед передачей модели
        x_t = x_t.to(device)
        
        # Предсказываем шум с помощью модели
        model_output = model(x_t, t, **processed_kwargs)
        
        # Вычисляем потери в зависимости от типа модели
        if self.model_mean_type == "epsilon":
            # Для модели предсказания шума: MSE между шумом и предсказанием
            target = noise
        elif self.model_mean_type == "direct":
            # Для модели прямого предсказания: MSE между x_0 и предсказанием
            target = x_start
        else:
            raise ValueError(f"Неизвестный model_mean_type: {self.model_mean_type}")
        
        # Убедимся, что target и model_output на одном устройстве
        target = target.to(device)
        model_output = model_output.to(device)
        
        # Вычисляем потери
        if self.loss_type == "mse":
            # Среднеквадратичная ошибка
            loss = ((model_output - target) ** 2).mean(dim=list(range(1, len(target.shape))))
        elif self.loss_type == "kl":
            # Kullback-Leibler дивергенция
            # В данной реализации не используется
            raise NotImplementedError("KL divergence loss не реализован")
        else:
            raise ValueError(f"Неизвестный loss_type: {self.loss_type}")
        
        return {"loss": loss, "model_output": model_output, "target": target}


def extract_and_expand(arr: np.ndarray, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Извлекает значения из массива по индексам t и расширяет их до формы x
    
    Args:
        arr: исходный массив numpy
        t: тензор с индексами
        x: тензор, форму которого нужно использовать
        
    Returns:
        Тензор с извлеченными значениями, расширенный до формы x
    """
    device = t.device
    
    # Убедимся, что t и x находятся на одном устройстве
    x = x.to(device)
    
    # Преобразуем массив numpy в тензор и перемещаем его на нужное устройство
    arr_torch = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    
    # Индексируем массив
    out = arr_torch.gather(-1, t.long())
    
    # Расширяем размерность до формы x
    while len(out.shape) < len(x.shape):
        out = out.unsqueeze(-1)
    
    # Расширяем тензор до формы x
    return out.expand(x.shape)


def get_beta_schedule(
    schedule: str = "linear", 
    num_timesteps: int = 1000, 
    beta_start: float = 1e-4, 
    beta_end: float = 2e-2
) -> np.ndarray:
    """
    Создает расписание коэффициентов beta для процесса диффузии
    
    Args:
        schedule: тип расписания ("linear", "cosine")
        num_timesteps: количество шагов
        beta_start: начальное значение beta
        beta_end: конечное значение beta
        
    Returns:
        Массив с коэффициентами beta
    """
    if schedule == "linear":
        # Линейное расписание
        return np.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "cosine":
        # Косинусное расписание
        steps = num_timesteps + 1
        x = np.linspace(0, num_timesteps, steps)
        alphas_cumprod = np.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Неизвестный schedule: {schedule}") 