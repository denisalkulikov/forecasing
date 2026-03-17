from nicegui import ui
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import asyncio
import warnings
import os
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT")
}

class ForecastApp:
    def __init__(self):
        self.df_years = pd.DataFrame()
        self.df_sales = pd.DataFrame()
        self.monthly_df = None
        self.weekly_df = None
        self.daily_df = None
        self.years_table = None
        self.forecast_table = None
        self.progress_bar = None
        self.progress_text = None
        self.create_ui()
    
    def get_db_connection(self):
        """Создание подключения к БД"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            ui.notify(f'Ошибка подключения к БД: {e}', type='negative')
            return None
    
    def load_years_data(self):
        """Загрузка данных по годам из division_results"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            select year, 
                   sum(total_amount_plan) as plan, 
                   sum(total_amount_summary) as fact
            from kamtent.division_results
            group by year
            order by year
            """
            df = pd.read_sql(query, conn)
            
            # Округляем до 2 знаков после запятой
            for col in ['plan', 'fact']:
                if col in df.columns:
                    df[col] = df[col].round(2)
            
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по годам: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()
    
    def load_sales_data(self):
        """Загрузка данных о продажах для прогнозирования"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                pay_date,
                pay_summ as sales_sum
            FROM kamtent.sales
            WHERE pay_date IS NOT NULL
            ORDER BY pay_date
            """
            df = pd.read_sql(query, conn)
            df['pay_date'] = pd.to_datetime(df['pay_date'])
            df['sales_sum'] = df['sales_sum'].round(2)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных о продажах: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()
    
    def prepare_monthly_data(self, df):
        """Подготовка месячных данных"""
        df = df.copy()
        df['year'] = df['pay_date'].dt.year
        df['month'] = df['pay_date'].dt.month
        
        # Агрегация по месяцам
        monthly = df.groupby(['year', 'month'])['sales_sum'].sum().reset_index()
        monthly['Date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        monthly = monthly[['Date', 'sales_sum']].rename(columns={'sales_sum': 'Sales'})
        monthly = monthly.set_index('Date').sort_index()
        
        # Интерполяция пропусков
        monthly['Sales'] = monthly['Sales'].interpolate(method='linear')
        monthly['Sales'] = monthly['Sales'].bfill().ffill()
        
        return monthly
    
    def prepare_weekly_data(self, df):
        """Подготовка недельных данных"""
        df = df.copy()
        df = df.set_index('pay_date').sort_index()
        
        # Агрегация по неделям (начиная с понедельника)
        weekly = df.resample('W-MON')['sales_sum'].sum()
        weekly = weekly.to_frame(name='Sales')
        
        # Интерполяция пропусков
        weekly['Sales'] = weekly['Sales'].interpolate(method='linear')
        weekly['Sales'] = weekly['Sales'].bfill().ffill()
        
        return weekly
    
    def prepare_daily_data(self, df):
        """Подготовка ежедневных данных"""
        df = df.copy()
        df = df.set_index('pay_date').sort_index()
        
        # Агрегация по дням
        daily = df.resample('D')['sales_sum'].sum()
        daily = daily.to_frame(name='Sales')
        
        # Интерполяция пропусков
        daily['Sales'] = daily['Sales'].interpolate(method='time')
        daily['Sales'] = daily['Sales'].bfill().ffill()
        
        return daily
    
    def round_amount(self, amount, precision='hundreds_thousands'):
        """Округление суммы до нужной точности"""
        if precision == 'hundreds_thousands':
            # Округление до сотен тысяч (100 000)
            return round(amount / 100000) * 100000
        elif precision == 'tens_thousands':
            # Округление до десятков тысяч (10 000)
            return round(amount / 10000) * 10000
        elif precision == 'thousands':
            # Округление до тысяч (1 000)
            return round(amount / 1000) * 1000
        else:
            # Округление до сотых (0.01)
            return round(amount, 2)
    
    async def update_progress(self, value, text):
        """Обновление прогресс-бара"""
        if self.progress_bar:
            self.progress_bar.set_value(value)
        if self.progress_text:
            self.progress_text.set_text(text)
        await asyncio.sleep(0.1)  # Небольшая задержка для обновления UI
    
    def forecast_for_year(self, data, agg_level, target_year, min_monthly=2100000):
        """Прогнозирование на указанный год с разными уровнями агрегации"""
        
        if agg_level == 'monthly':
            return self.forecast_monthly(data, target_year, min_monthly)
        elif agg_level == 'weekly':
            return self.forecast_weekly_optimized(data, target_year, min_monthly)
        elif agg_level == 'daily':
            return self.forecast_daily(data, target_year, min_monthly)
        else:
            return None
    
    def forecast_monthly(self, df, target_year, min_monthly):
        """Прогноз на основе месячных данных (упрощенный)"""
        # Разделение на train/test
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 12:  # Уменьшаем требование до 1 года
            return self.fallback_forecast(train, target_year, min_monthly, 'monthly')
        
        models = {}
        
        # Используем только простые модели для надежности
        
        # Простое скользящее среднее (всегда работает)
        try:
            last_12 = train.tail(12)['Sales'].values
            if len(last_12) >= 6:
                # Используем среднее за последние 6 месяцев с учетом сезонности
                seasonal_factors = []
                for i in range(1, 13):
                    same_month_data = train[train.index.month == i]['Sales'].tail(3)
                    if len(same_month_data) > 0:
                        seasonal_factors.append(same_month_data.mean() / train['Sales'].tail(12).mean())
                    else:
                        seasonal_factors.append(1.0)
                
                # Нормализуем факторы
                seasonal_factors = np.array(seasonal_factors)
                seasonal_factors = seasonal_factors / seasonal_factors.mean() * 1.0
                
                base = train['Sales'].tail(6).mean()
                forecast = [base * factor for factor in seasonal_factors]
                models['Seasonal_MA'] = forecast
        except Exception as e:
            print(f"Seasonal MA failed: {e}")
        
        # Holt-Winters (упрощенный)
        try:
            train_values = train['Sales'].bfill().ffill()
            if len(train_values) >= 12:
                hw_model = ExponentialSmoothing(
                    train_values, 
                    trend='add', 
                    seasonal='add',
                    seasonal_periods=12, 
                    damped_trend=True
                )
                hw_fit = hw_model.fit()
                hw_forecast = hw_fit.forecast(12)
                models['Holt-Winters'] = hw_forecast
        except Exception as e:
            print(f"Holt-Winters failed: {e}")
        
        # Prophet (упрощенный)
        try:
            prophet_df = train.reset_index()[['Date', 'Sales']]
            prophet_df.columns = ['ds', 'y']
            prophet_df['y'] = prophet_df['y'].bfill().ffill()
            if len(prophet_df) >= 12:
                prophet_model = Prophet(
                    seasonality_mode='multiplicative', 
                    yearly_seasonality=True,
                    seasonality_prior_scale=0.1
                )
                prophet_model.fit(prophet_df)
                future = prophet_model.make_future_dataframe(periods=12, freq='ME')
                prophet_forecast = prophet_model.predict(future)['yhat'][-12:].values
                models['Prophet'] = prophet_forecast
        except Exception as e:
            print(f"Prophet failed: {e}")
        
        # Если есть хотя бы одна модель, используем её
        if models:
            if len(models) > 1:
                # Если есть несколько моделей, используем среднее
                ensemble = pd.DataFrame(models).mean(axis=1)
                forecast = ensemble
                model_name = 'Ensemble'
            else:
                # Используем единственную модель
                model_name = list(models.keys())[0]
                forecast = models[model_name]
            
            # Применяем минимальный порог и округление
            forecast = [max(float(x), min_monthly) for x in forecast]
        else:
            # Если все модели не сработали, используем fallback
            return self.fallback_forecast(train, target_year, min_monthly, 'monthly')
        
        # Создаем DataFrame с прогнозом
        forecast_dates = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        forecast_df = pd.DataFrame({
            'month': forecast_dates,
            'forecast': forecast
        })
        
        # Добавляем статистику с округлением
        stats = {
            'total_forecast': self.round_amount(sum(forecast_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(forecast_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(forecast_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(forecast_df['forecast']), 'hundreds_thousands'),
            'model_used': model_name
        }
        
        # Округляем сам прогноз
        forecast_df['forecast'] = forecast_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        return forecast_df, stats
    
    def forecast_weekly_optimized(self, df, target_year, min_monthly):
        """Оптимизированный прогноз на основе недельных данных"""
        
        # Разделение на train/test
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 26:  # Уменьшаем требование до 26 недель (полгода)
            return self.fallback_forecast(train, target_year, min_monthly, 'weekly')
        
        models = {}
        
        # Используем только Holt-Winters и Prophet (SARIMA слишком тяжелая для недельных данных)
        
        # Holt-Winters (оптимизированный)
        try:
            train_values = train['Sales'].bfill().ffill()
            # Упрощаем модель для ускорения
            hw_model = ExponentialSmoothing(
                train_values, 
                trend='add', 
                seasonal='add',
                seasonal_periods=13,  # Уменьшаем период до 13 недель (квартал)
                damped_trend=True
            )
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(26)  # Прогноз на 26 недель (полгода)
            models['Holt-Winters'] = hw_forecast
        except Exception as e:
            print(f"Holt-Winters failed: {e}")
        
        # Prophet (оптимизированный)
        try:
            prophet_df = train.reset_index().rename(columns={'pay_date': 'ds', 'Sales': 'y'})
            prophet_df['y'] = prophet_df['y'].bfill().ffill()
            # Упрощаем модель Prophet
            prophet_model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,  # Отключаем недельную сезонность для ускорения
                seasonality_prior_scale=0.1  # Уменьшаем сложность
            )
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=26, freq='W-MON')
            prophet_forecast = prophet_model.predict(future)['yhat'][-26:].values
            models['Prophet'] = prophet_forecast
        except Exception as e:
            print(f"Prophet failed: {e}")
        
        # Если есть хотя бы одна модель, используем её
        if models:
            if len(models) > 1:
                # Если есть несколько моделей, используем среднее
                ensemble = pd.DataFrame(models).mean(axis=1)
                weekly_forecast = ensemble
                model_name = 'Ensemble'
            else:
                # Используем единственную модель
                model_name = list(models.keys())[0]
                weekly_forecast = models[model_name]
        else:
            return self.fallback_forecast(train, target_year, min_monthly, 'weekly')
        
        # Интерполируем прогноз до 52 недель
        if len(weekly_forecast) < 52:
            # Создаем индексы для интерполяции
            x_old = np.linspace(0, 1, len(weekly_forecast))
            x_new = np.linspace(0, 1, 52)
            weekly_forecast_full = np.interp(x_new, x_old, weekly_forecast)
        else:
            weekly_forecast_full = weekly_forecast[:52]
        
        # Конвертируем недельный прогноз в месячный
        forecast_dates = pd.date_range(start=f'{target_year}-01-05', periods=52, freq='W-MON')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': weekly_forecast_full
        })
        
        # Агрегация по месяцам
        forecast_df.set_index('Date', inplace=True)
        monthly_forecast = forecast_df.resample('ME')['Forecast'].sum()
        
        # Применяем минимальный порог к месяцам
        monthly_forecast = [max(x, min_monthly) for x in monthly_forecast]
        
        forecast_dates_monthly = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates_monthly,
            'forecast': monthly_forecast
        })
        
        # Округляем до сотен тысяч
        result_df['forecast'] = result_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        stats = {
            'total_forecast': self.round_amount(sum(result_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(result_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(result_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(result_df['forecast']), 'hundreds_thousands'),
            'model_used': f"{model_name} (weekly-based)"
        }
        
        return result_df, stats
    
    def forecast_daily(self, df, target_year, min_monthly):
        """Прогноз на основе ежедневных данных с агрегацией в месячные"""
        # Определяем количество дней в году
        days_in_year = 366 if (target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0)) else 365
        
        # Разделение на train/test
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 180:  # Уменьшаем требование до 180 дней (полгода)
            return self.fallback_forecast(train, target_year, min_monthly, 'daily')
        
        models = {}
        
        # Используем только Prophet для дневных данных (остальные слишком тяжелые)
        
        # Prophet
        try:
            prophet_df = train.reset_index()[['pay_date','Sales']].rename(
                columns={'pay_date':'ds','Sales':'y'}
            )
            prophet_df['y'] = prophet_df['y'].bfill().ffill()
            prophet_model = Prophet(
                seasonality_mode='multiplicative', 
                yearly_seasonality=True, 
                weekly_seasonality=True,
                seasonality_prior_scale=0.1
            )
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=min(days_in_year, 90), freq='D')  # Ограничиваем прогноз 90 днями
            prophet_forecast = prophet_model.predict(future)['yhat'][-min(days_in_year, 90):].values
            models['Prophet'] = prophet_forecast
        except Exception as e:
            print(f"Prophet failed: {e}")
        
        # Если есть модель, используем её
        if models:
            daily_forecast = models['Prophet']
            model_name = 'Prophet'
            
            # Если прогноз короче, чем нужно, интерполируем
            if len(daily_forecast) < days_in_year:
                x_old = np.linspace(0, 1, len(daily_forecast))
                x_new = np.linspace(0, 1, days_in_year)
                daily_forecast = np.interp(x_new, x_old, daily_forecast)
        else:
            return self.fallback_forecast(train, target_year, min_monthly, 'daily')
        
        # Конвертируем дневной прогноз в месячный
        forecast_dates = pd.date_range(start=f'{target_year}-01-01', periods=days_in_year, freq='D')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': daily_forecast[:days_in_year]
        })
        
        # Агрегация по месяцам
        forecast_df.set_index('Date', inplace=True)
        monthly_forecast = forecast_df.resample('ME')['Forecast'].sum()
        
        # Применяем минимальный порог к месяцам
        monthly_forecast = [max(x, min_monthly) for x in monthly_forecast]
        
        forecast_dates_monthly = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates_monthly,
            'forecast': monthly_forecast
        })
        
        # Округляем до сотен тысяч
        result_df['forecast'] = result_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        stats = {
            'total_forecast': self.round_amount(sum(result_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(result_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(result_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(result_df['forecast']), 'hundreds_thousands'),
            'model_used': f"{model_name} (daily-based)"
        }
        
        return result_df, stats
    
    def fallback_forecast(self, train, target_year, min_monthly, agg_level):
        """Запасной метод прогнозирования, если основные модели не работают"""
        if len(train) > 0:
            # Используем среднее за последние 12 месяцев
            last_values = train.tail(12)['Sales']
            avg_monthly = last_values.mean() if len(last_values) > 0 else min_monthly * 1.5
        else:
            avg_monthly = min_monthly * 1.5
        
        # Применяем минимальный порог
        monthly_forecast = [max(avg_monthly, min_monthly)] * 12
        
        forecast_dates = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates,
            'forecast': monthly_forecast
        })
        
        # Округляем до сотен тысяч
        result_df['forecast'] = result_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        stats = {
            'total_forecast': self.round_amount(sum(result_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(result_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(result_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(result_df['forecast']), 'hundreds_thousands'),
            'model_used': f"Fallback ({agg_level})"
        }
        
        return result_df, stats
    
    def select_best_model(self, models, test_data):
        """Выбор лучшей модели на основе MAE на тестовых данных"""
        best_mae = float('inf')
        best_model = 'Ensemble'
        
        for name, forecast in models.items():
            if name == 'Ensemble':
                continue
            
            # Сравниваем с соответствующим периодом тестовых данных
            test_period = test_data.iloc[:len(forecast)]
            
            if len(test_period) > 0 and len(forecast) > 0:
                try:
                    mae = mean_absolute_error(test_period['Sales'], forecast[:len(test_period)])
                    if mae < best_mae:
                        best_mae = mae
                        best_model = name
                except:
                    pass
        
        return best_model
    
    def update_years_table(self):
        """Обновление таблицы с данными по годам"""
        self.df_years = self.load_years_data()
        
        # Очищаем контейнер и создаем новую таблицу
        years_container.clear()
        
        if self.df_years.empty:
            with years_container:
                ui.label('Нет данных по годам').classes('text-bold text-red')
            return
        
        with years_container:
            ui.label('Данные по годам:').classes('text-h6 text-bold mb-2')
            
            columns = [
                {'name': 'year', 'label': 'Год', 'field': 'year', 'align': 'center'},
                {'name': 'plan', 'label': 'План (₽)', 'field': 'plan', 'align': 'right'},
                {'name': 'fact', 'label': 'Факт (₽)', 'field': 'fact', 'align': 'right'},
            ]
            
            rows = self.df_years.to_dict('records')
            
            # Округляем значения в таблице
            for row in rows:
                row['plan'] = self.round_amount(row['plan'], 'hundreds_thousands')
                row['fact'] = self.round_amount(row['fact'], 'hundreds_thousands')
            
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key='year',
                pagination={'rowsPerPage': 10}
            ).classes('w-full')
            
            # Форматирование чисел
            table.add_slot('body-cell-year', '''
                <q-td key="year" :props="props" style="width: 80px; min-width: 80px;">
                    <div class="text-center text-bold">{{ props.value }}</div>
                </q-td>
            ''')
            
            table.add_slot('body-cell-plan', '''
                <q-td key="plan" :props="props" style="text-align: right;">
                    <div class="font-mono">
                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                    </div>
                </q-td>
            ''')
            
            table.add_slot('body-cell-fact', '''
                <q-td key="fact" :props="props" style="text-align: right;">
                    <div class="font-mono">
                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                    </div>
                </q-td>
            ''')
            
            self.years_table = table
    
    async def on_forecast_click(self):
        """Обработчик кнопки прогноза (асинхронный)"""
        selected_year = self.select_year.value
        agg_level = self.agg_select.value
        
        # Показываем прогресс-бар
        progress_container.clear()
        with progress_container:
            with ui.row().classes('items-center gap-4'):
                self.progress_bar = ui.circular_progress(
                    value=0, 
                    min=0, 
                    max=100, 
                    size='50px',
                    show_value=False
                )
                self.progress_text = ui.label('Начинаем расчет...').classes('text-bold')
        
        ui.notify(f'Расчет прогноза на {selected_year} год ({agg_level})...', type='info')
        
        await self.update_progress(10, 'Загрузка данных...')
        
        # Загружаем данные, если еще не загружены
        if self.df_sales.empty:
            self.df_sales = self.load_sales_data()
            if not self.df_sales.empty:
                self.monthly_df = self.prepare_monthly_data(self.df_sales)
                self.weekly_df = self.prepare_weekly_data(self.df_sales)
                self.daily_df = self.prepare_daily_data(self.df_sales)
        
        if self.df_sales.empty:
            forecast_container.clear()
            with forecast_container:
                ui.label('Нет данных для прогнозирования').classes('text-bold text-red')
            progress_container.clear()
            return
        
        await self.update_progress(20, 'Подготовка данных...')
        
        # Восстанавливаем таблицу с годами (если она была очищена)
        if self.df_years.empty:
            self.update_years_table()
        
        # Выбираем соответствующие данные
        if agg_level == 'По месяцам':
            data = self.monthly_df
            agg = 'monthly'
            await self.update_progress(30, 'Месячный прогноз...')
        elif agg_level == 'По неделям':
            data = self.weekly_df
            agg = 'weekly'
            await self.update_progress(30, 'Недельный прогноз (может занять до 30 секунд)...')
        else:  # По дням
            data = self.daily_df
            agg = 'daily'
            await self.update_progress(30, 'Дневной прогноз...')
        
        # Делаем прогноз
        try:
            forecast_result = self.forecast_for_year(
                data, 
                agg, 
                selected_year,
                min_monthly=2100000
            )
        except Exception as e:
            print(f"Ошибка прогноза: {e}")
            forecast_result = None
        
        await self.update_progress(80, 'Формирование результатов...')
        
        # Очищаем только контейнер прогноза
        forecast_container.clear()
        
        if forecast_result is None:
            with forecast_container:
                ui.label('Ошибка при расчете прогноза').classes('text-bold text-red')
        else:
            forecast_df, stats = forecast_result
            
            with forecast_container:
                ui.label(f'ПРОГНОЗ НА {selected_year} ГОД').classes('text-h5 text-bold text-blue mb-4')
                
                # Информация о модели
                ui.label(f"Модель: {stats['model_used']}").classes('text-italic mb-2')
                
                # Карточки с общей статистикой (уже округлено)
                with ui.row().classes('w-full gap-4 mb-6'):
                    with ui.card().classes('bg-blue-1 p-4'):
                        ui.label('Общая сумма:').classes('text-bold')
                        ui.label(f"{stats['total_forecast']:,.0f} ₽".replace(',', ' ')).classes('text-h6 text-bold text-blue')
                    
                    with ui.card().classes('bg-green-1 p-4'):
                        ui.label('Среднемесячно:').classes('text-bold')
                        ui.label(f"{stats['avg_monthly']:,.0f} ₽".replace(',', ' ')).classes('text-h6 text-bold text-green')
                    
                    with ui.card().classes('bg-purple-1 p-4'):
                        ui.label('Мин/Макс месяц:').classes('text-bold')
                        ui.label(f"{stats['min_month']:,.0f} / {stats['max_month']:,.0f} ₽".replace(',', ' ')).classes('text-h6 text-bold text-purple')
                
                # Таблица с помесячным прогнозом
                ui.label('Помесячный прогноз:').classes('text-h6 text-bold mb-2')
                
                columns = [
                    {'name': 'month', 'label': 'Месяц', 'field': 'month', 'align': 'left'},
                    {'name': 'forecast', 'label': 'Прогноз (₽)', 'field': 'forecast', 'align': 'right'},
                ]
                
                months_ru = {
                    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
                    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
                    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
                }
                
                rows = []
                for _, row in forecast_df.iterrows():
                    month_num = row['month'].month
                    rows.append({
                        'month': f"{months_ru[month_num]} {row['month'].year}",
                        'forecast': row['forecast']
                    })
                
                forecast_table = ui.table(
                    columns=columns,
                    rows=rows,
                    pagination={'rowsPerPage': 12}
                ).classes('w-full')
                
                forecast_table.add_slot('body-cell-forecast', '''
                    <q-td key="forecast" :props="props" style="text-align: right;">
                        <div class="font-mono text-bold">
                            {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                        </div>
                    </q-td>
                ''')
                
                self.forecast_table = forecast_table
        
        await self.update_progress(100, 'Готово!')
        
        # Убираем прогресс-бар через 2 секунды
        await asyncio.sleep(2)
        progress_container.clear()
    
    def on_load_all_click(self):
        """Загрузка всех данных"""
        ui.notify('Загрузка данных...', type='info')
        self.update_years_table()
        self.df_sales = self.load_sales_data()
        
        if not self.df_sales.empty:
            self.monthly_df = self.prepare_monthly_data(self.df_sales)
            self.weekly_df = self.prepare_weekly_data(self.df_sales)
            self.daily_df = self.prepare_daily_data(self.df_sales)
            ui.notify(f'Загружено {len(self.df_sales)} записей', type='positive')
        else:
            ui.notify('Нет данных о продажах', type='warning')
        
        # Очищаем прогноз при загрузке новых данных
        forecast_container.clear()
    
    def create_ui(self):
        """Создание интерфейса"""
        # Добавляем CSS
        ui.add_head_html('''
        <style>
        .font-mono {
            font-family: 'Courier New', monospace !important;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        .q-table td, .q-table th {
            padding: 8px 12px !important;
        }
        .bg-blue-1 { background-color: #e3f2fd; }
        .bg-green-1 { background-color: #e8f5e8; }
        .bg-purple-1 { background-color: #f3e5f5; }
        </style>
        ''')
        
        ui.label('Анализ и прогнозирование данных').classes('text-h3 text-bold text-center w-full mb-6')
        
        # Панель управления
        with ui.row().classes('justify-center w-full gap-4 mb-6'):
            ui.button('Загрузить данные', on_click=self.on_load_all_click).classes('bg-blue-500 text-white px-8 py-2')
            
            self.select_year = ui.select(
                [2020, 2021, 2022, 2023, 2024, 2025, 2026],
                label='Выберите год для прогноза',
                value=2026
            ).classes('w-48')
            
            self.agg_select = ui.select(
                ['По месяцам', 'По неделям', 'По дням'],
                label='Уровень агрегации',
                value='По месяцам'
            ).classes('w-48')
            
            ui.button('Сделать прогноз', on_click=self.on_forecast_click).classes('bg-green-500 text-white px-6 py-2')
        
        # Контейнер для прогресс-бара
        global progress_container
        progress_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')
        
        # Контейнер для данных по годам
        global years_container
        years_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4 mb-8')
        
        # Контейнер для прогноза
        global forecast_container
        forecast_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

# Запуск приложения
app = ForecastApp()

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=True)