import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """
    Кастомная среда для симуляции торгов на фондовом рынке.
    Реализует логику MDP (Марковского процесса принятия решений).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, config):
        super(StockTradingEnv, self).__init__()
        
        # --- Параметры Конфигурации ---
        self.df = df # Весь датасет с признаками
        self.cfg = config['model_light']
        self.prog_cfg = config['progress']
        
        self.n_stocks = self.cfg['n_stocks']
        self.n_days_history = self.cfg['n_days_history']
        self.start_capital = self.cfg['start_capital']
        
        # Комиссии (экранированные значения согласно Принципу №7)
        self.comm_buy = self.cfg['comission_on_buy']
        self.comm_sell = self.cfg['comission_on_sale']
        self.comm_day = self.cfg['comission_on_new_day']

        # --- Определение Пространств (Spaces) ---
        
        # Action Space: Доля капитала для каждого стока (-1 до 1)
        # Мы используем Multi-head Attention, поэтому агент будет выдавать веса для N бумаг
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        # Observation Space: [Кол-во акций, Окно истории, Кол-во признаков]
        # Примечание: Мы добавляем +1 к признакам для передачи мета-данных портфеля
        n_features = len([
            'open', 'high', 'low', 'close', 'volume', 'adj_close', 
            'close_mean_3d', 'close_std_3d', 'close_mean_30d', 'close_std_30d',
            'volume_mean_7d', 'log_return_1d', 'price_range_1d'
        ])
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_stocks, self.n_days_history, n_features), 
            dtype=np.float32
        )

        # --- Внутренние переменные состояния ---
        self.current_step = 0
        self.balance = self.start_capital
        self.shares_held = np.zeros(self.n_stocks)
        self.history_variance = [] # Для логики выхода по дисперсии

    def reset(self, seed=None, options=None):
        """
        Сброс среды к начальному состоянию. 
        Вызывается в начале каждого эпизода обучения.
        """
        super().reset(seed=seed)
        
        # Инциализация баланса и шагов
        self.balance = self.start_capital
        self.current_step = self.n_days_history
        self.shares_held = np.zeros(self.n_stocks)
        self.history_variance = []
        
        # Выбор случайных акций для эпизода (Lighter Model constraint)
        self.active_symbols = np.random.choice(
            self.df['symbol'].unique(), self.n_stocks, replace=False
        )
        
        observation = self._get_obs()
        info = {} # Доп. информация для отладки
        
        return observation, info

    def step(self, action):
        """
        Переход среды из состояния S в S+1 под воздействием действия A.
        """
        # 1. Текущие цены закрытия (для оценки портфеля)
        current_prices = self._get_current_prices()
        
        # 2. Логика исполнения ордеров (Execution)
        # Действие агента интерпретируем как желаемую аллокацию портфеля
        self._trade(action, current_prices)
        
        # 3. Переход на следующий временной шаг
        self.current_step += 1
        
        # 4. Применение комиссии за перенос позиции (Overnight/New day)
        self.balance -= self.balance * self.comm_day
        
        # 5. Расчет вознаграждения (Reward)
        # В SAC важно, чтобы Reward был масштабирован. Используем логарифмическую доходность.
        new_net_worth = self._get_net_worth(current_prices)
        reward = np.log(new_net_worth / (self.last_net_worth + 1e-8))
        self.last_net_worth = new_net_worth
        
        # 6. Проверка условий завершения (Termination)
        terminated = self.current_step >= (len(self.df) // self.n_stocks) - 1
        
        # Кастомная логика выхода по дисперсии (variance из конфига)
        self.history_variance.append(new_net_worth)
        truncated = False
        if len(self.history_variance) >= self.cfg['n_tries']:
            recent_var = np.var(self.history_variance[-self.cfg['n_tries']:])
            if recent_var < self.cfg['variance']:
                truncated = True # Ранняя остановка, если эквити "умерло"

        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, {"net_worth": new_net_worth}

    def _get_obs(self):
        """
        Формирует тензор наблюдений для Transformer модели.
        Извлекает срез данных по каждой активной акции за n_days_history.
        """
        obs_matrix = []
        for symbol in self.active_symbols:
            # Извлекаем окно данных для конкретного тикера
            stock_data = self.df[self.df['symbol'] == symbol].iloc[
                self.current_step - self.n_days_history : self.current_step
            ]
            # Удаляем нечисловые колонки (date, symbol)
            features = stock_data.drop(columns=['symbol', 'date']).values
            obs_matrix.append(features)
            
        return np.array(obs_matrix, dtype=np.float32)

    def _trade(self, action, prices):
        """
        Механика покупки/продажи. Обновляет self.balance и self.shares_held.
        """
        for i, weight in enumerate(action):
            # weight > 0: Хотим купить/держать
            # weight < 0: Хотим продать
            if weight > 0:
                # Сколько максимально можем купить на выделенную долю
                amount_to_spend = self.balance * weight
                shares_to_buy = amount_to_spend / (prices[i] * (1 + self.comm_buy))
                self.shares_held[i] += shares_to_buy
                self.balance -= amount_to_spend
            elif weight < 0:
                # Продаем процент от имеющихся акций
                shares_to_sell = self.shares_held[i] * abs(weight)
                gain = shares_to_sell * prices[i] * (1 - self.comm_sell)
                self.shares_held[i] -= shares_to_sell
                self.balance += gain

    def _get_net_worth(self, prices):
        return self.balance + np.sum(self.shares_held * prices)

    def _get_current_prices(self):
        # Получение цен закрытия всех активных акций на текущий шаг
        prices = []
        for symbol in self.active_symbols:
            price = self.df[self.df['symbol'] == symbol].iloc[self.current_step]['close']
            prices.append(price)
        return np.array(prices)