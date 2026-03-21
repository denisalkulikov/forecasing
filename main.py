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

# Загружаем переменные окружения из файла .env
load_dotenv()

warnings.filterwarnings('ignore')

# Конфигурация БД из переменных окружения
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": os.getenv("DB_PORT", "5432")
}

class ForecastApp:
    def __init__(self):
        self.df_years = pd.DataFrame()
        self.df_sales = pd.DataFrame()
        self.df_directions = pd.DataFrame()
        self.df_oai_groups = pd.DataFrame()
        self.df_kn_groups = pd.DataFrame()
        self.df_reklama_groups = pd.DataFrame()
        self.df_tk_groups = pd.DataFrame()
        self.direction_coefficients = {}
        self.oai_group_coefficients = {}
        self.kn_group_coefficients = {}
        self.reklama_group_coefficients = {}
        self.tk_group_coefficients = {}
        self.monthly_df = None
        self.weekly_df = None
        self.daily_df = None
        self.years_table = None
        self.forecast_table = None
        self.direction_forecast_table = None
        self.oai_group_forecast_table = None
        self.kn_group_forecast_table = None
        self.reklama_group_forecast_table = None
        self.tk_group_forecast_table = None
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
        """Загрузка данных по годам из division_results (сохраняем копейки)"""
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
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по годам: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()
    
    def load_directions_data(self):
        """Загрузка данных по товарным направлениям"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                year,
                direction,
                total_amount_actual as actual,
                total_amount_plan as plan
            FROM kamtent.yearly_division_results
            ORDER BY year, direction
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по направлениям: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()
    
    def calculate_direction_coefficients(self, target_year):
        """Расчет средних коэффициентов по направлениям на основе всех годов до target_year - 2"""
        if self.df_directions.empty:
            self.df_directions = self.load_directions_data()
        
        if self.df_directions.empty:
            return {}
        
        # Берем все годы, которые меньше или равны target_year - 2
        max_year = target_year - 2
        years = sorted([y for y in self.df_directions['year'].unique() if y <= max_year])
        
        # Если данных нет, берем последние доступные годы
        if not years:
            years = sorted(self.df_directions['year'].unique())[-3:]
        
        print(f"\n{'='*60}")
        print(f"Расчет средних коэффициентов на основе годов: {years}")
        print(f"(все годы до {max_year}, включительно)")
        print(f"{'='*60}")
        
        # Собираем коэффициенты по каждому году
        all_coefficients = {}
        
        for year in years:
            df_year = self.df_directions[self.df_directions['year'] == year]
            year_totals = df_year.groupby('direction')['actual'].sum()
            year_total = year_totals.sum()
            
            if year_total > 0:
                print(f"\n{year} год:")
                print("-" * 40)
                for direction, total in year_totals.items():
                    coef = total / year_total
                    print(f"  {direction}: {coef:.6f} ({coef*100:.2f}%)")
                    
                    if direction not in all_coefficients:
                        all_coefficients[direction] = []
                    all_coefficients[direction].append(coef)
        
        # Рассчитываем средние коэффициенты
        coefficients = {}
        print(f"\n{'='*60}")
        print(f"СРЕДНИЕ КОЭФФИЦИЕНТЫ за {len(years)} лет ({years[0]}-{years[-1]}):")
        print(f"{'='*60}")
        
        for direction, coef_list in all_coefficients.items():
            avg_coef = sum(coef_list) / len(coef_list)
            coefficients[direction] = avg_coef
            print(f"  {direction}: {avg_coef:.6f} ({avg_coef*100:.2f}%)")
            print(f"    Диапазон: {min(coef_list):.6f} - {max(coef_list):.6f}")
        
        # Выводим общую сумму коэффициентов для проверки
        total_coef = sum(coefficients.values())
        print(f"\nСумма коэффициентов: {total_coef:.6f}")
        
        # Нормализуем, если сумма не равна 1
        if abs(total_coef - 1.0) > 0.0001:
            print(f"ВНИМАНИЕ: Сумма коэффициентов = {total_coef:.6f}, выполняем нормализацию...")
            for direction in coefficients:
                coefficients[direction] = coefficients[direction] / total_coef
            
            print("\nПосле нормализации:")
            for direction, coef in coefficients.items():
                print(f"  {direction}: {coef:.6f} ({coef*100:.2f}%)")
        
        return coefficients
    
    def split_forecast_by_directions(self, monthly_forecast, target_year):
        """Разбивка помесячного прогноза по направлениям"""
        coefficients = self.calculate_direction_coefficients(target_year)
        
        if not coefficients:
            ui.notify('Нет данных по направлениям для разбивки прогноза', type='warning')
            return None
        
        # Создаем DataFrame с разбивкой по месяцам и направлениям
        result = []
        
        for _, row in monthly_forecast.iterrows():
            month = row['month']
            total = row['forecast']
            
            # Разбиваем без округления
            forecast_values = {}
            for direction, coef in coefficients.items():
                forecast_values[direction] = total * coef
            
            # Округляем до тысяч, но не теряем малые суммы
            rounded_values = {}
            remaining = total
            
            # Сначала округляем все направления, кроме последнего
            directions_list = list(coefficients.keys())
            for i, direction in enumerate(directions_list[:-1]):
                rounded = max(1000, round(forecast_values[direction] / 1000) * 1000)
                rounded_values[direction] = rounded
                remaining -= rounded
            
            # Последнее направление получает остаток
            last_direction = directions_list[-1]
            rounded_values[last_direction] = max(0, remaining)
            
            # Добавляем в результат
            for direction, value in rounded_values.items():
                result.append({
                    'month': month,
                    'direction': direction,
                    'forecast': value
                })
        
        result_df = pd.DataFrame(result)
        
        # Выводим проверку сходимости
        print("\nПроверка сходимости сумм после корректировки:")
        for month_num in range(1, 13):
            month_total = monthly_forecast[monthly_forecast['month'].dt.month == month_num]['forecast'].sum()
            directions_sum = result_df[result_df['month'].dt.month == month_num]['forecast'].sum()
            print(f"  Месяц {month_num}: прогноз={month_total:,.0f}, сумма по направлениям={directions_sum:,.0f}, разница={month_total - directions_sum:,.0f}")
        
        return result_df
    
    def load_oai_group_data(self):
        """Загрузка данных по товарным группам для направления ОАИ"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                year,
                month,
                direction,
                group_product,
                pay_summ
            FROM kamtent.monthly_group_product
            WHERE direction = 'ОАИ'
            ORDER BY year, month, group_product
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по товарным группам: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()

    def calculate_oai_group_coefficients(self, target_year):
        """Расчет коэффициентов для товарных групп ОАИ на основе всех годов до target_year - 2"""
        if self.df_oai_groups.empty:
            self.df_oai_groups = self.load_oai_group_data()
        
        if self.df_oai_groups.empty:
            return {}
        
        # Берем все годы, которые меньше или равны target_year - 2
        max_year = target_year - 2
        years = sorted([y for y in self.df_oai_groups['year'].unique() if y <= max_year])
        
        # Если данных нет, берем последние доступные годы
        if not years:
            years = sorted(self.df_oai_groups['year'].unique())[-3:]
        
        print(f"\n{'='*60}")
        print(f"Расчет коэффициентов для товарных групп ОАИ на основе годов: {years}")
        print(f"(все годы до {max_year}, включительно)")
        print(f"{'='*60}")
        
        # Собираем коэффициенты по каждому году
        all_coefficients = {}
        
        for year in years:
            df_year = self.df_oai_groups[self.df_oai_groups['year'] == year]
            # Суммируем по месяцам для каждого года
            year_totals = df_year.groupby('group_product')['pay_summ'].sum()
            year_total = year_totals.sum()
            
            if year_total > 0:
                print(f"\n{year} год:")
                print("-" * 40)
                for group, total in year_totals.items():
                    coef = total / year_total
                    print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
                    
                    if group not in all_coefficients:
                        all_coefficients[group] = []
                    all_coefficients[group].append(coef)
        
        # Рассчитываем средние коэффициенты
        coefficients = {}
        print(f"\n{'='*60}")
        print(f"СРЕДНИЕ КОЭФФИЦИЕНТЫ для товарных групп ОАИ за {len(years)} лет ({years[0]}-{years[-1]}):")
        print(f"{'='*60}")
        
        for group, coef_list in all_coefficients.items():
            avg_coef = sum(coef_list) / len(coef_list)
            coefficients[group] = avg_coef
            print(f"  {group}: {avg_coef:.6f} ({avg_coef*100:.2f}%)")
            print(f"    Диапазон: {min(coef_list):.6f} - {max(coef_list):.6f}")
        
        # Выводим общую сумму коэффициентов для проверки
        total_coef = sum(coefficients.values())
        print(f"\nСумма коэффициентов: {total_coef:.6f}")
        
        # Нормализуем, если сумма не равна 1
        if abs(total_coef - 1.0) > 0.0001:
            print(f"ВНИМАНИЕ: Сумма коэффициентов = {total_coef:.6f}, выполняем нормализацию...")
            for group in coefficients:
                coefficients[group] = coefficients[group] / total_coef
            
            print("\nПосле нормализации:")
            for group, coef in coefficients.items():
                print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
        
        return coefficients

    def split_oai_by_groups(self, oai_forecast, target_year):
        """Разбивка годового прогноза ОАИ по товарным группам"""
        coefficients = self.calculate_oai_group_coefficients(target_year)
        
        if not coefficients:
            ui.notify('Нет данных по товарным группам для разбивки прогноза ОАИ', type='warning')
            return None
        
        # Считаем общую сумму ОАИ за год
        total_oai = oai_forecast['forecast'].sum()
        
        print(f"\n{'='*60}")
        print(f"Разбивка годового прогноза ОАИ ({total_oai:,.0f} ₽) по товарным группам:")
        print(f"{'='*60}")
        
        # Разбиваем общую сумму по группам
        result = []
        for group, coef in coefficients.items():
            group_value = total_oai * coef
            # Округляем до тысяч
            group_value = max(1000, round(group_value / 1000) * 1000)
            result.append({
                'group': group,
                'forecast': group_value
            })
        
        result_df = pd.DataFrame(result)
        
        # Корректируем сумму, если есть расхождение
        total_result = result_df['forecast'].sum()
        diff = total_oai - total_result
        
        if diff != 0:
            print(f"Расхождение: {diff:,.0f} ₽, корректируем наибольшую группу")
            # Находим группу с максимальным значением и корректируем её
            max_idx = result_df['forecast'].idxmax()
            result_df.loc[max_idx, 'forecast'] = result_df.loc[max_idx, 'forecast'] + diff
        
        # Выводим результат
        print("\nРезультат разбивки:")
        for _, row in result_df.iterrows():
            print(f"  {row['group']}: {row['forecast']:,.0f} ₽ ({row['forecast']/total_oai*100:.1f}%)")
        
        return result_df
    
    def load_kn_groups_data(self):
        """Загрузка данных по товарным группам для направления КН"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                year,
                month,
                direction,
                group_product,
                pay_summ
            FROM kamtent.monthly_group_product
            WHERE direction = 'КН'
            ORDER BY year, month, group_product
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по товарным группам КН: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()

    def calculate_kn_group_coefficients(self, target_year):
        """Расчет коэффициентов для товарных групп КН на основе всех годов до target_year - 2"""
        if self.df_kn_groups.empty:
            self.df_kn_groups = self.load_kn_groups_data()
        
        if self.df_kn_groups.empty:
            return {}
        
        # Берем все годы, которые меньше или равны target_year - 2
        max_year = target_year - 2
        years = sorted([y for y in self.df_kn_groups['year'].unique() if y <= max_year])
        
        # Если данных нет, берем последние доступные годы
        if not years:
            years = sorted(self.df_kn_groups['year'].unique())[-3:]
        
        print(f"\n{'='*60}")
        print(f"Расчет коэффициентов для товарных групп КН на основе годов: {years}")
        print(f"(все годы до {max_year}, включительно)")
        print(f"{'='*60}")
        
        # Собираем коэффициенты по каждому году
        all_coefficients = {}
        
        for year in years:
            df_year = self.df_kn_groups[self.df_kn_groups['year'] == year]
            # Суммируем по месяцам для каждого года
            year_totals = df_year.groupby('group_product')['pay_summ'].sum()
            year_total = year_totals.sum()
            
            if year_total > 0:
                print(f"\n{year} год:")
                print("-" * 40)
                for group, total in year_totals.items():
                    coef = total / year_total
                    print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
                    
                    if group not in all_coefficients:
                        all_coefficients[group] = []
                    all_coefficients[group].append(coef)
        
        # Рассчитываем средние коэффициенты
        coefficients = {}
        print(f"\n{'='*60}")
        print(f"СРЕДНИЕ КОЭФФИЦИЕНТЫ для товарных групп КН за {len(years)} лет ({years[0]}-{years[-1]}):")
        print(f"{'='*60}")
        
        for group, coef_list in all_coefficients.items():
            avg_coef = sum(coef_list) / len(coef_list)
            coefficients[group] = avg_coef
            print(f"  {group}: {avg_coef:.6f} ({avg_coef*100:.2f}%)")
            print(f"    Диапазон: {min(coef_list):.6f} - {max(coef_list):.6f}")
        
        # Выводим общую сумму коэффициентов для проверки
        total_coef = sum(coefficients.values())
        print(f"\nСумма коэффициентов: {total_coef:.6f}")
        
        # Нормализуем, если сумма не равна 1
        if abs(total_coef - 1.0) > 0.0001:
            print(f"ВНИМАНИЕ: Сумма коэффициентов = {total_coef:.6f}, выполняем нормализацию...")
            for group in coefficients:
                coefficients[group] = coefficients[group] / total_coef
            
            print("\nПосле нормализации:")
            for group, coef in coefficients.items():
                print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
        
        return coefficients

    def split_kn_by_groups(self, kn_forecast, target_year):
        """Разбивка годового прогноза КН по товарным группам"""
        coefficients = self.calculate_kn_group_coefficients(target_year)
        
        if not coefficients:
            ui.notify('Нет данных по товарным группам для разбивки прогноза КН', type='warning')
            return None
        
        # Считаем общую сумму КН за год
        total_kn = kn_forecast['forecast'].sum()
        
        print(f"\n{'='*60}")
        print(f"Разбивка годового прогноза КН ({total_kn:,.0f} ₽) по товарным группам:")
        print(f"{'='*60}")
        
        # Разбиваем общую сумму по группам
        result = []
        for group, coef in coefficients.items():
            group_value = total_kn * coef
            # Округляем до тысяч
            group_value = max(1000, round(group_value / 1000) * 1000)
            result.append({
                'group': group,
                'forecast': group_value
            })
        
        result_df = pd.DataFrame(result)
        
        # Корректируем сумму, если есть расхождение
        total_result = result_df['forecast'].sum()
        diff = total_kn - total_result
        
        if diff != 0:
            print(f"Расхождение: {diff:,.0f} ₽, корректируем наибольшую группу")
            # Находим группу с максимальным значением и корректируем её
            max_idx = result_df['forecast'].idxmax()
            result_df.loc[max_idx, 'forecast'] = result_df.loc[max_idx, 'forecast'] + diff
        
        # Выводим результат
        print("\nРезультат разбивки:")
        for _, row in result_df.iterrows():
            print(f"  {row['group']}: {row['forecast']:,.0f} ₽ ({row['forecast']/total_kn*100:.1f}%)")
        
        return result_df
    
    def load_reklama_groups_data(self):
        """Загрузка данных по товарным группам для направления РЕКЛАМА"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                year,
                month,
                direction,
                group_product,
                pay_summ
            FROM kamtent.monthly_group_product
            WHERE direction = 'РЕКЛАМА'
            ORDER BY year, month, group_product
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по товарным группам РЕКЛАМА: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()

    def calculate_reklama_group_coefficients(self, target_year):
        """Расчет коэффициентов для товарных групп РЕКЛАМА на основе всех годов до target_year - 2"""
        if self.df_reklama_groups.empty:
            self.df_reklama_groups = self.load_reklama_groups_data()
        
        if self.df_reklama_groups.empty:
            return {}
        
        # Берем все годы, которые меньше или равны target_year - 2
        max_year = target_year - 2
        years = sorted([y for y in self.df_reklama_groups['year'].unique() if y <= max_year])
        
        # Если данных нет, берем последние доступные годы
        if not years:
            years = sorted(self.df_reklama_groups['year'].unique())[-3:]
        
        print(f"\n{'='*60}")
        print(f"Расчет коэффициентов для товарных групп РЕКЛАМА на основе годов: {years}")
        print(f"(все годы до {max_year}, включительно)")
        print(f"{'='*60}")
        
        # Собираем коэффициенты по каждому году
        all_coefficients = {}
        
        for year in years:
            df_year = self.df_reklama_groups[self.df_reklama_groups['year'] == year]
            # Суммируем по месяцам для каждого года
            year_totals = df_year.groupby('group_product')['pay_summ'].sum()
            year_total = year_totals.sum()
            
            if year_total > 0:
                print(f"\n{year} год:")
                print("-" * 40)
                for group, total in year_totals.items():
                    coef = total / year_total
                    print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
                    
                    if group not in all_coefficients:
                        all_coefficients[group] = []
                    all_coefficients[group].append(coef)
        
        # Рассчитываем средние коэффициенты
        coefficients = {}
        print(f"\n{'='*60}")
        print(f"СРЕДНИЕ КОЭФФИЦИЕНТЫ для товарных групп РЕКЛАМА за {len(years)} лет ({years[0]}-{years[-1]}):")
        print(f"{'='*60}")
        
        for group, coef_list in all_coefficients.items():
            avg_coef = sum(coef_list) / len(coef_list)
            coefficients[group] = avg_coef
            print(f"  {group}: {avg_coef:.6f} ({avg_coef*100:.2f}%)")
            print(f"    Диапазон: {min(coef_list):.6f} - {max(coef_list):.6f}")
        
        # Выводим общую сумму коэффициентов для проверки
        total_coef = sum(coefficients.values())
        print(f"\nСумма коэффициентов: {total_coef:.6f}")
        
        # Нормализуем, если сумма не равна 1
        if abs(total_coef - 1.0) > 0.0001:
            print(f"ВНИМАНИЕ: Сумма коэффициентов = {total_coef:.6f}, выполняем нормализацию...")
            for group in coefficients:
                coefficients[group] = coefficients[group] / total_coef
            
            print("\nПосле нормализации:")
            for group, coef in coefficients.items():
                print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
        
        return coefficients

    def split_reklama_by_groups(self, reklama_forecast, target_year):
        """Разбивка годового прогноза РЕКЛАМА по товарным группам"""
        coefficients = self.calculate_reklama_group_coefficients(target_year)
        
        if not coefficients:
            ui.notify('Нет данных по товарным группам для разбивки прогноза РЕКЛАМА', type='warning')
            return None
        
        # Считаем общую сумму КН за год
        total_reklama = reklama_forecast['forecast'].sum()
        
        print(f"\n{'='*60}")
        print(f"Разбивка годового прогноза РЕКЛАМА ({total_reklama:,.0f} ₽) по товарным группам:")
        print(f"{'='*60}")
        
        # Разбиваем общую сумму по группам
        result = []
        for group, coef in coefficients.items():
            group_value = total_reklama * coef
            # Округляем до тысяч
            group_value = max(1000, round(group_value / 1000) * 1000)
            result.append({
                'group': group,
                'forecast': group_value
            })
        
        result_df = pd.DataFrame(result)
        
        # Корректируем сумму, если есть расхождение
        total_result = result_df['forecast'].sum()
        diff = total_reklama - total_result
        
        if diff != 0:
            print(f"Расхождение: {diff:,.0f} ₽, корректируем наибольшую группу")
            # Находим группу с максимальным значением и корректируем её
            max_idx = result_df['forecast'].idxmax()
            result_df.loc[max_idx, 'forecast'] = result_df.loc[max_idx, 'forecast'] + diff
        
        # Выводим результат
        print("\nРезультат разбивки:")
        for _, row in result_df.iterrows():
            print(f"  {row['group']}: {row['forecast']:,.0f} ₽ ({row['forecast']/total_reklama*100:.1f}%)")
        
        return result_df
    
    def load_tk_groups_data(self):
        """Загрузка данных по товарным группам для направления ТК"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                year,
                month,
                direction,
                group_product,
                pay_summ
            FROM kamtent.monthly_group_product
            WHERE direction = 'ТК'
            ORDER BY year, month, group_product
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            ui.notify(f'Ошибка загрузки данных по товарным группам ТК: {e}', type='negative')
            return pd.DataFrame()
        finally:
            conn.close()

    def calculate_tk_group_coefficients(self, target_year):
        """Расчет коэффициентов для товарных групп ТК на основе всех годов до target_year - 2"""
        if self.df_tk_groups.empty:
            self.df_tk_groups = self.load_tk_groups_data()
        
        if self.df_tk_groups.empty:
            return {}
        
        # Группируем товарные группы по новым категориям
        group_mapping = {
            'ТОРГОВЫЕ ТК': 'Торговые ТК',
            'ПРОМЫШЛЕННЫЕ ТК': 'Промышленные ТК',
            'СПОРТИВНЫЕ И КУЛЬТ. ТК': 'Спортивные и культ. ТК',
            'СЕЛЬСКОХОЗЯЙСТВЕННЫЕ ТК': 'Сельскохозяйственные ТК',
            'ПРОЧЕЕ': 'Прочее',
            'ОРИГИНАЛЬНЫЕ ТК': 'Прочее'
        }
        
        # Применяем группировку
        self.df_tk_groups['group_category'] = self.df_tk_groups['group_product'].map(group_mapping)
        
        # Берем все годы, которые меньше или равны target_year - 2
        max_year = target_year - 2
        years = sorted([y for y in self.df_tk_groups['year'].unique() if y <= max_year])
        
        # Если данных нет, берем последние доступные годы
        if not years:
            years = sorted(self.df_tk_groups['year'].unique())[-3:]
        
        print(f"\n{'='*60}")
        print(f"Расчет коэффициентов для товарных групп ТК на основе годов: {years}")
        print(f"(все годы до {max_year}, включительно)")
        print(f"{'='*60}")
        
        # Собираем коэффициенты по каждому году
        all_coefficients = {}
        
        for year in years:
            df_year = self.df_tk_groups[self.df_tk_groups['year'] == year]
            
            # Группируем по новым категориям
            year_totals = df_year.groupby('group_category')['pay_summ'].sum()
            
            # Добавляем фиксированную категорию "Строители (пологи/шторы)ТК" с нулевой суммой
            # Она будет обработана отдельно
            if 'Строители (пологи/шторы)ТК' not in year_totals.index:
                year_totals['Строители (пологи/шторы)ТК'] = 0
            
            year_total = year_totals.sum()
            
            if year_total > 0:
                print(f"\n{year} год:")
                print("-" * 40)
                for group, total in year_totals.items():
                    coef = total / year_total
                    print(f"  {group}: {coef:.6f} ({coef*100:.2f}%) - сумма: {total:,.2f}")
                    
                    if group not in all_coefficients:
                        all_coefficients[group] = []
                    all_coefficients[group].append(coef)
        
        # Рассчитываем средние коэффициенты
        coefficients = {}
        print(f"\n{'='*60}")
        print(f"СРЕДНИЕ КОЭФФИЦИЕНТЫ для товарных групп ТК за {len(years)} лет ({years[0]}-{years[-1]}):")
        print(f"{'='*60}")
        
        for group, coef_list in all_coefficients.items():
            avg_coef = sum(coef_list) / len(coef_list)
            coefficients[group] = avg_coef
            print(f"  {group}: {avg_coef:.6f} ({avg_coef*100:.2f}%)")
            print(f"    Диапазон: {min(coef_list):.6f} - {max(coef_list):.6f}")
        
        # Выводим общую сумму коэффициентов для проверки
        total_coef = sum(coefficients.values())
        print(f"\nСумма коэффициентов: {total_coef:.6f}")
        
        # Нормализуем, если сумма не равна 1
        if abs(total_coef - 1.0) > 0.0001:
            print(f"ВНИМАНИЕ: Сумма коэффициентов = {total_coef:.6f}, выполняем нормализацию...")
            for group in coefficients:
                coefficients[group] = coefficients[group] / total_coef
            
            print("\nПосле нормализации:")
            for group, coef in coefficients.items():
                print(f"  {group}: {coef:.6f} ({coef*100:.2f}%)")
        
        return coefficients

    def split_tk_by_groups(self, tk_forecast, target_year):
        """Разбивка годового прогноза ТК по товарным группам"""
        coefficients = self.calculate_tk_group_coefficients(target_year)
        
        if not coefficients:
            ui.notify('Нет данных по товарным группам для разбивки прогноза ТК', type='warning')
            return None
        
        # Считаем общую сумму ТК за год
        total_tk = tk_forecast['forecast'].sum()
        
        print(f"\n{'='*60}")
        print(f"Разбивка годового прогноза ТК ({total_tk:,.0f} ₽) по товарным группам:")
        print(f"{'='*60}")
        
        # Разбиваем общую сумму по группам
        result = []
        for group, coef in coefficients.items():
            if group == 'Строители (пологи/шторы)ТК':
                # Фиксированная сумма 50,000 для этой группы
                group_value = 50000
            else:
                group_value = total_tk * coef
                # Округляем до тысяч
                group_value = max(1000, round(group_value / 1000) * 1000)
            result.append({
                'group': group,
                'forecast': group_value
            })
        
        result_df = pd.DataFrame(result)
        
        # Корректируем сумму, если есть расхождение (исключая фиксированную группу)
        total_result = result_df[result_df['group'] != 'Строители (пологи/шторы)ТК']['forecast'].sum()
        total_result += 50000  # Добавляем фиксированную сумму
        
        diff = total_tk - total_result
        
        if diff != 0:
            print(f"Расхождение: {diff:,.0f} ₽, корректируем наибольшую группу")
            # Находим группу с максимальным значением (кроме фиксированной)
            max_group = result_df[result_df['group'] != 'Строители (пологи/шторы)ТК'].loc[result_df['forecast'].idxmax()]['group']
            max_idx = result_df[result_df['group'] == max_group].index[0]
            result_df.loc[max_idx, 'forecast'] = result_df.loc[max_idx, 'forecast'] + diff
        
        # Выводим результат
        print("\nРезультат разбивки:")
        for _, row in result_df.iterrows():
            print(f"  {row['group']}: {row['forecast']:,.0f} ₽ ({row['forecast']/total_tk*100:.1f}%)")
        
        return result_df
    
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
            return round(amount / 100000) * 100000
        elif precision == 'tens_thousands':
            return round(amount / 10000) * 10000
        elif precision == 'thousands':
            return round(amount / 1000) * 1000
        else:
            return round(amount, 2)
    
    async def update_progress(self, value, text):
        """Обновление прогресс-бара"""
        if self.progress_bar:
            self.progress_bar.set_value(value)
        if self.progress_text:
            self.progress_text.set_text(text)
        await asyncio.sleep(0.1)
    
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
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 12:
            return self.fallback_forecast(train, target_year, min_monthly, 'monthly')
        
        models = {}
        
        # Простое скользящее среднее
        try:
            last_12 = train.tail(12)['Sales'].values
            if len(last_12) >= 6:
                seasonal_factors = []
                for i in range(1, 13):
                    same_month_data = train[train.index.month == i]['Sales'].tail(3)
                    if len(same_month_data) > 0:
                        seasonal_factors.append(same_month_data.mean() / train['Sales'].tail(12).mean())
                    else:
                        seasonal_factors.append(1.0)
                
                seasonal_factors = np.array(seasonal_factors)
                seasonal_factors = seasonal_factors / seasonal_factors.mean() * 1.0
                
                base = train['Sales'].tail(6).mean()
                forecast = [base * factor for factor in seasonal_factors]
                models['Seasonal_MA'] = forecast
        except Exception as e:
            print(f"Seasonal MA failed: {e}")
        
        # Holt-Winters
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
        
        # Prophet
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
        
        if models:
            if len(models) > 1:
                ensemble = pd.DataFrame(models).mean(axis=1)
                forecast = ensemble
                model_name = 'Ensemble'
            else:
                model_name = list(models.keys())[0]
                forecast = models[model_name]
            
            forecast = [max(float(x), min_monthly) for x in forecast]
        else:
            return self.fallback_forecast(train, target_year, min_monthly, 'monthly')
        
        forecast_dates = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        forecast_df = pd.DataFrame({
            'month': forecast_dates,
            'forecast': forecast
        })
        
        forecast_df['forecast'] = forecast_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        stats = {
            'total_forecast': self.round_amount(sum(forecast_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(forecast_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(forecast_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(forecast_df['forecast']), 'hundreds_thousands'),
            'model_used': model_name
        }
        
        return forecast_df, stats
    
    def forecast_weekly_optimized(self, df, target_year, min_monthly):
        """Оптимизированный прогноз на основе недельных данных"""
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 26:
            return self.fallback_forecast(train, target_year, min_monthly, 'weekly')
        
        models = {}
        
        # Holt-Winters
        try:
            train_values = train['Sales'].bfill().ffill()
            hw_model = ExponentialSmoothing(
                train_values, 
                trend='add', 
                seasonal='add',
                seasonal_periods=13,
                damped_trend=True
            )
            hw_fit = hw_model.fit()
            hw_forecast = hw_fit.forecast(26)
            models['Holt-Winters'] = hw_forecast
        except Exception as e:
            print(f"Holt-Winters failed: {e}")
        
        # Prophet
        try:
            prophet_df = train.reset_index().rename(columns={'pay_date': 'ds', 'Sales': 'y'})
            prophet_df['y'] = prophet_df['y'].bfill().ffill()
            prophet_model = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                seasonality_prior_scale=0.1
            )
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=26, freq='W-MON')
            prophet_forecast = prophet_model.predict(future)['yhat'][-26:].values
            models['Prophet'] = prophet_forecast
        except Exception as e:
            print(f"Prophet failed: {e}")
        
        if models:
            if len(models) > 1:
                ensemble = pd.DataFrame(models).mean(axis=1)
                weekly_forecast = ensemble
                model_name = 'Ensemble'
            else:
                model_name = list(models.keys())[0]
                weekly_forecast = models[model_name]
        else:
            return self.fallback_forecast(train, target_year, min_monthly, 'weekly')
        
        # Проверяем длину прогноза
        print(f"Длина прогноза: {len(weekly_forecast)}")
        
        # Если прогноз короче 52 недель, интерполируем
        if len(weekly_forecast) < 52:
            x_old = np.linspace(0, 1, len(weekly_forecast))
            x_new = np.linspace(0, 1, 52)
            try:
                weekly_forecast_full = np.interp(x_new, x_old, weekly_forecast)
            except Exception as e:
                print(f"Ошибка интерполяции: {e}")
                # Если интерполяция не работает, просто повторяем последнее значение
                weekly_forecast_full = np.full(52, weekly_forecast[-1])
        elif len(weekly_forecast) > 52:
            weekly_forecast_full = weekly_forecast[:52]
        else:
            weekly_forecast_full = weekly_forecast
        
        # Убеждаемся, что длина равна 52
        if len(weekly_forecast_full) != 52:
            print(f"Некорректная длина прогноза: {len(weekly_forecast_full)}")
            # Заполняем до 52 недель
            if len(weekly_forecast_full) < 52:
                weekly_forecast_full = np.append(weekly_forecast_full, [weekly_forecast_full[-1]] * (52 - len(weekly_forecast_full)))
            else:
                weekly_forecast_full = weekly_forecast_full[:52]
        
        # Создаем даты для прогноза
        forecast_dates = pd.date_range(start=f'{target_year}-01-05', periods=52, freq='W-MON')
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': weekly_forecast_full})
        forecast_df.set_index('Date', inplace=True)
        
        # Агрегируем по месяцам
        monthly_forecast = forecast_df.resample('ME')['Forecast'].sum()
        
        # Применяем минимальный порог
        monthly_forecast = [max(x, min_monthly) for x in monthly_forecast]
        
        # Убеждаемся, что у нас 12 месяцев
        if len(monthly_forecast) < 12:
            monthly_forecast = monthly_forecast + [monthly_forecast[-1]] * (12 - len(monthly_forecast))
        elif len(monthly_forecast) > 12:
            monthly_forecast = monthly_forecast[:12]
        
        forecast_dates_monthly = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates_monthly,
            'forecast': monthly_forecast
        })
        
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
        days_in_year = 366 if (target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0)) else 365
        train = df[df.index <= f'{target_year-1}-12-31']
        
        if len(train) < 180:
            return self.fallback_forecast(train, target_year, min_monthly, 'daily')
        
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
            future = prophet_model.make_future_dataframe(periods=min(days_in_year, 90), freq='D')
            daily_forecast = prophet_model.predict(future)['yhat'][-min(days_in_year, 90):].values
            
            if len(daily_forecast) < days_in_year:
                x_old = np.linspace(0, 1, len(daily_forecast))
                x_new = np.linspace(0, 1, days_in_year)
                daily_forecast = np.interp(x_new, x_old, daily_forecast)
        except Exception as e:
            print(f"Prophet failed: {e}")
            return self.fallback_forecast(train, target_year, min_monthly, 'daily')
        
        forecast_dates = pd.date_range(start=f'{target_year}-01-01', periods=days_in_year, freq='D')
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': daily_forecast[:days_in_year]})
        forecast_df.set_index('Date', inplace=True)
        monthly_forecast = forecast_df.resample('ME')['Forecast'].sum()
        monthly_forecast = [max(x, min_monthly) for x in monthly_forecast]
        
        forecast_dates_monthly = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates_monthly,
            'forecast': monthly_forecast
        })
        
        result_df['forecast'] = result_df['forecast'].apply(
            lambda x: self.round_amount(x, 'hundreds_thousands')
        )
        
        stats = {
            'total_forecast': self.round_amount(sum(result_df['forecast']), 'hundreds_thousands'),
            'avg_monthly': self.round_amount(np.mean(result_df['forecast']), 'hundreds_thousands'),
            'min_month': self.round_amount(min(result_df['forecast']), 'hundreds_thousands'),
            'max_month': self.round_amount(max(result_df['forecast']), 'hundreds_thousands'),
            'model_used': "Prophet (daily-based)"
        }
        
        return result_df, stats
    
    def fallback_forecast(self, train, target_year, min_monthly, agg_level):
        """Запасной метод прогнозирования"""
        if len(train) > 0:
            last_values = train.tail(12)['Sales']
            avg_monthly = last_values.mean() if len(last_values) > 0 else min_monthly * 1.5
        else:
            avg_monthly = min_monthly * 1.5
        
        monthly_forecast = [max(avg_monthly, min_monthly)] * 12
        forecast_dates = pd.date_range(start=f'{target_year}-01-31', periods=12, freq='ME')
        
        result_df = pd.DataFrame({
            'month': forecast_dates,
            'forecast': monthly_forecast
        })
        
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
    
    def update_years_table(self):
        """Обновление таблицы с данными по годам"""
        self.df_years = self.load_years_data()
        
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
            
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key='year',
                pagination={'rowsPerPage': 10}
            ).classes('w-full')
            
            table.add_slot('body-cell-year', '''
                <q-td key="year" :props="props" style="width: 80px; min-width: 80px; text-align: center;">
                    <div class="text-bold">{{ props.value }}</div>
                </q-td>
            ''')
            
            table.add_slot('body-cell-plan', '''
                <q-td key="plan" :props="props" style="text-align: right;">
                    <div class="font-mono">
                        {{ new Intl.NumberFormat('ru-RU', {minimumFractionDigits: 2, maximumFractionDigits: 2}).format(props.value) }}
                    </div>
                </q-td>
            ''')
            
            table.add_slot('body-cell-fact', '''
                <q-td key="fact" :props="props" style="text-align: right;">
                    <div class="font-mono">
                        {{ new Intl.NumberFormat('ru-RU', {minimumFractionDigits: 2, maximumFractionDigits: 2}).format(props.value) }}
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
            direction_container.clear()
            with forecast_container:
                ui.label('Нет данных для прогнозирования').classes('text-bold text-red')
            progress_container.clear()
            return
        
        await self.update_progress(20, 'Подготовка данных...')
        
        # Загружаем данные по направлениям
        if self.df_directions.empty:
            self.df_directions = self.load_directions_data()
        
        await self.update_progress(25, 'Расчет коэффициентов по направлениям...')
        
        # Восстанавливаем таблицу с годами
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
        else:
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
        
        await self.update_progress(70, 'Формирование результатов...')
        
        # Очищаем контейнеры
        forecast_container.clear()
        direction_container.clear()
        
        if forecast_result is None:
            with forecast_container:
                ui.label('Ошибка при расчете прогноза').classes('text-bold text-red')
        else:
            forecast_df, stats = forecast_result
            
            # Разбиваем прогноз по направлениям
            direction_forecast = self.split_forecast_by_directions(forecast_df, selected_year)
            
            # Отображаем общий прогноз
            with forecast_container:
                ui.label(f'ПРОГНОЗ НА {selected_year} ГОД').classes('text-h5 text-bold text-blue mb-4')
                ui.label(f"Модель: {stats['model_used']}").classes('text-italic mb-2')
                
                # Карточки со статистикой
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
                
                # Таблица помесячного прогноза
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
                        'month': months_ru[month_num],
                        'forecast': row['forecast']
                    })
                
                forecast_table = ui.table(
                    columns=columns,
                    rows=rows,
                    pagination={'rowsPerPage': 15}
                ).classes('w-full')
                
                forecast_table.add_slot('body-cell-month', '''
                    <q-td key="month" :props="props" style="text-align: left;">
                        <div class="text-bold">
                            {{ props.value }}
                        </div>
                    </q-td>
                ''')
                
                forecast_table.add_slot('body-cell-forecast', '''
                    <q-td key="forecast" :props="props" style="text-align: right;">
                        <div class="font-mono text-bold">
                            {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                        </div>
                    </q-td>
                ''')
            
            # Отображаем прогноз по направлениям
            if direction_forecast is not None and not direction_forecast.empty:
                with direction_container:
                    ui.label('ПРОГНОЗ ПО ТОВАРНЫМ НАПРАВЛЕНИЯМ:').classes('text-h5 text-bold text-green mb-4')
                    
                    # Получаем уникальные направления
                    directions = sorted(direction_forecast['direction'].unique())
                    
                    # Создаем колонки для таблицы
                    columns = [{'name': 'month', 'label': 'Месяц', 'field': 'month', 'align': 'left'}]
                    for direction in directions:
                        columns.append({
                            'name': direction,
                            'label': direction,
                            'field': direction,
                            'align': 'right'
                        })
                    
                    # Формируем строки
                    months_ru = {
                        1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
                        5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
                        9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
                    }
                    
                    rows = []
                    for month_num in range(1, 13):
                        row = {'month': months_ru[month_num]}
                        month_data = direction_forecast[direction_forecast['month'].dt.month == month_num]
                        for direction in directions:
                            value = month_data[month_data['direction'] == direction]['forecast'].sum()
                            row[direction] = value if value > 0 else 0
                        rows.append(row)
                    
                    # Добавляем итоговую строку
                    total_row = {'month': 'ИТОГО:'}
                    for direction in directions:
                        total = direction_forecast[direction_forecast['direction'] == direction]['forecast'].sum()
                        total_row[direction] = total
                    rows.append(total_row)
                    
                    # Создаем таблицу
                    direction_table = ui.table(
                        columns=columns,
                        rows=rows,
                        pagination={'rowsPerPage': 15}
                    ).classes('w-full')
                    
                    # Добавляем форматирование для колонки месяца
                    direction_table.add_slot('body-cell-month', '''
                        <q-td key="month" :props="props" style="text-align: left;">
                            <div class="text-bold">
                                {{ props.value }}
                            </div>
                        </q-td>
                    ''')
                    
                    # Добавляем форматирование для всех числовых колонок
                    for direction in directions:
                        # Используем двойные фигурные скобки для экранирования в f-string
                        direction_table.add_slot(f'body-cell-{direction}', f'''
                            <q-td key="{direction}" :props="props" style="text-align: right;">
                                <div class="font-mono text-bold">
                                    {{{{ new Intl.NumberFormat('ru-RU', {{maximumFractionDigits: 0}}).format(props.value) }}}}
                                </div>
                            </q-td>
                        ''')
                    
                    # Обновляем таблицу с итоговой строкой
                    direction_table.rows = rows

            # После отображения прогноза по направлениям, добавляем разбивку ОАИ
            if direction_forecast is not None and not direction_forecast.empty:
                # Получаем прогноз только для ОАИ
                oai_forecast = direction_forecast[direction_forecast['direction'] == 'ОАИ'][['month', 'forecast']]
                if not oai_forecast.empty:
                    # Разбиваем ОАИ по товарным группам
                    oai_groups_forecast = self.split_oai_by_groups(oai_forecast, selected_year)
                    
                    if oai_groups_forecast is not None and not oai_groups_forecast.empty:
                        with oai_group_container:
                            ui.label('ПРОГНОЗ ПО ТОВАРНЫМ ГРУППАМ ОАИ:').classes('text-h5 text-bold text-orange mb-4')
                            
                            # Получаем уникальные группы
                            groups = sorted(oai_groups_forecast['group'].unique())
                            
                            # Создаем колонки для таблицы
                            columns = [
                                {'name': 'group', 'label': 'Товарная группа', 'field': 'group', 'align': 'left'},
                                {'name': 'forecast', 'label': 'Прогноз (₽)', 'field': 'forecast', 'align': 'right'},
                                {'name': 'percentage', 'label': 'Доля (%)', 'field': 'percentage', 'align': 'right'}
                            ]
                            
                            # Формируем строки
                            total_oai = oai_groups_forecast['forecast'].sum()
                            rows = []
                            for _, row in oai_groups_forecast.iterrows():
                                percentage = (row['forecast'] / total_oai) * 100 if total_oai > 0 else 0
                                rows.append({
                                    'group': row['group'],
                                    'forecast': row['forecast'],
                                    'percentage': percentage
                                })
                            
                            # Добавляем итоговую строку
                            rows.append({
                                'group': 'ИТОГО:',
                                'forecast': total_oai,
                                'percentage': 100.0
                            })
                            
                            # Создаем таблицу
                            group_table = ui.table(
                                columns=columns,
                                rows=rows,
                                pagination={'rowsPerPage': 20}
                            ).classes('w-full')
                            
                            # Добавляем форматирование для колонки группы
                            group_table.add_slot('body-cell-group', '''
                                <q-td key="group" :props="props" style="text-align: left;">
                                    <div class="text-bold">
                                        {{ props.value }}
                                    </div>
                                </q-td>
                            ''')
                            
                            # Добавляем форматирование для прогноза
                            group_table.add_slot('body-cell-forecast', '''
                                <q-td key="forecast" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                                    </div>
                                </q-td>
                            ''')
                            
                            # Добавляем форматирование для процентов
                            group_table.add_slot('body-cell-percentage', '''
                                <q-td key="percentage" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ props.value.toFixed(1) }}%
                                    </div>
                                </q-td>
                            ''')
                            
                            group_table.rows = rows

            # После разбивки ОАИ, добавляем разбивку КН
            if direction_forecast is not None and not direction_forecast.empty:
                # Получаем прогноз только для КН
                kn_forecast = direction_forecast[direction_forecast['direction'] == 'КН'][['month', 'forecast']]
                if not kn_forecast.empty:
                    # Разбиваем КН по товарным группам
                    kn_groups_forecast = self.split_kn_by_groups(kn_forecast, selected_year)
                    
                    if kn_groups_forecast is not None and not kn_groups_forecast.empty:
                        with kn_group_container:
                            ui.label('ПРОГНОЗ ПО ТОВАРНЫМ ГРУППАМ КН:').classes('text-h5 text-bold text-purple mb-4')
                            
                            # Получаем уникальные группы
                            kn_groups = sorted(kn_groups_forecast['group'].unique())
                            
                            # Создаем колонки для таблицы
                            columns = [
                                {'name': 'group', 'label': 'Товарная группа', 'field': 'group', 'align': 'left'},
                                {'name': 'forecast', 'label': 'Прогноз (₽)', 'field': 'forecast', 'align': 'right'},
                                {'name': 'percentage', 'label': 'Доля (%)', 'field': 'percentage', 'align': 'right'}
                            ]
                            
                            # Формируем строки
                            total_kn = kn_groups_forecast['forecast'].sum()
                            rows = []
                            for _, row in kn_groups_forecast.iterrows():
                                percentage = (row['forecast'] / total_kn) * 100 if total_kn > 0 else 0
                                rows.append({
                                    'group': row['group'],
                                    'forecast': row['forecast'],
                                    'percentage': percentage
                                })
                            
                            # Добавляем итоговую строку
                            rows.append({
                                'group': 'ИТОГО:',
                                'forecast': total_kn,
                                'percentage': 100.0
                            })
                            
                            # Создаем таблицу
                            kn_group_table = ui.table(
                                columns=columns,
                                rows=rows,
                                pagination={'rowsPerPage': 20}
                            ).classes('w-full')
                            
                            # Добавляем форматирование
                            kn_group_table.add_slot('body-cell-group', '''
                                <q-td key="group" :props="props" style="text-align: left;">
                                    <div class="text-bold">
                                        {{ props.value }}
                                    </div>
                                </q-td>
                            ''')
                            
                            kn_group_table.add_slot('body-cell-forecast', '''
                                <q-td key="forecast" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                                    </div>
                                </q-td>
                            ''')
                            
                            kn_group_table.add_slot('body-cell-percentage', '''
                                <q-td key="percentage" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ props.value.toFixed(1) }}%
                                    </div>
                                </q-td>
                            ''')
                            
                            kn_group_table.rows = rows

            # После разбивки ОАИ, добавляем разбивку РЕКЛАМА
            if direction_forecast is not None and not direction_forecast.empty:
                # Получаем прогноз только для КН
                reklama_forecast = direction_forecast[direction_forecast['direction'] == 'РЕКЛАМА'][['month', 'forecast']]
                if not reklama_forecast.empty:
                    # Разбиваем КН по товарным группам
                    reklama_groups_forecast = self.split_reklama_by_groups(reklama_forecast, selected_year)
                    
                    if reklama_groups_forecast is not None and not reklama_groups_forecast.empty:
                        with reklama_group_container:
                            ui.label('ПРОГНОЗ ПО ТОВАРНЫМ ГРУППАМ РЕКЛАМА:').classes('text-h5 text-bold text-purple mb-4')
                            
                            # Получаем уникальные группы
                            reklama_groups = sorted(reklama_groups_forecast['group'].unique())
                            
                            # Создаем колонки для таблицы
                            columns = [
                                {'name': 'group', 'label': 'Товарная группа', 'field': 'group', 'align': 'left'},
                                {'name': 'forecast', 'label': 'Прогноз (₽)', 'field': 'forecast', 'align': 'right'},
                                {'name': 'percentage', 'label': 'Доля (%)', 'field': 'percentage', 'align': 'right'}
                            ]
                            
                            # Формируем строки
                            total_reklama = reklama_groups_forecast['forecast'].sum()
                            rows = []
                            for _, row in reklama_groups_forecast.iterrows():
                                percentage = (row['forecast'] / total_reklama) * 100 if total_reklama > 0 else 0
                                rows.append({
                                    'group': row['group'],
                                    'forecast': row['forecast'],
                                    'percentage': percentage
                                })
                            
                            # Добавляем итоговую строку
                            rows.append({
                                'group': 'ИТОГО:',
                                'forecast': total_reklama,
                                'percentage': 100.0
                            })
                            
                            # Создаем таблицу
                            reklama_group_table = ui.table(
                                columns=columns,
                                rows=rows,
                                pagination={'rowsPerPage': 20}
                            ).classes('w-full')
                            
                            # Добавляем форматирование
                            reklama_group_table.add_slot('body-cell-group', '''
                                <q-td key="group" :props="props" style="text-align: left;">
                                    <div class="text-bold">
                                        {{ props.value }}
                                    </div>
                                </q-td>
                            ''')
                            
                            reklama_group_table.add_slot('body-cell-forecast', '''
                                <q-td key="forecast" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                                    </div>
                                </q-td>
                            ''')
                            
                            reklama_group_table.add_slot('body-cell-percentage', '''
                                <q-td key="percentage" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ props.value.toFixed(1) }}%
                                    </div>
                                </q-td>
                            ''')
                            
                            reklama_group_table.rows = rows

            # После разбивки КН, добавляем разбивку ТК
            if direction_forecast is not None and not direction_forecast.empty:
                # Получаем прогноз только для ТК
                tk_forecast = direction_forecast[direction_forecast['direction'] == 'ТК'][['month', 'forecast']]
                if not tk_forecast.empty:
                    # Разбиваем ТК по товарным группам
                    tk_groups_forecast = self.split_tk_by_groups(tk_forecast, selected_year)
                    
                    if tk_groups_forecast is not None and not tk_groups_forecast.empty:
                        with tk_group_container:
                            ui.label('ПРОГНОЗ ПО ТОВАРНЫМ ГРУППАМ ТК:').classes('text-h5 text-bold text-yellow mb-4')
                            
                            # Получаем уникальные группы
                            tk_groups = sorted(tk_groups_forecast['group'].unique())
                            
                            # Создаем колонки для таблицы
                            columns = [
                                {'name': 'group', 'label': 'Товарная группа', 'field': 'group', 'align': 'left'},
                                {'name': 'forecast', 'label': 'Прогноз (₽)', 'field': 'forecast', 'align': 'right'},
                                {'name': 'percentage', 'label': 'Доля (%)', 'field': 'percentage', 'align': 'right'}
                            ]
                            
                            # Формируем строки
                            total_tk = tk_groups_forecast['forecast'].sum()
                            rows = []
                            for _, row in tk_groups_forecast.iterrows():
                                percentage = (row['forecast'] / total_tk) * 100 if total_tk > 0 else 0
                                rows.append({
                                    'group': row['group'],
                                    'forecast': row['forecast'],
                                    'percentage': percentage
                                })
                            
                            # Добавляем итоговую строку
                            rows.append({
                                'group': 'ИТОГО:',
                                'forecast': total_tk,
                                'percentage': 100.0
                            })
                            
                            # Создаем таблицу
                            tk_group_table = ui.table(
                                columns=columns,
                                rows=rows,
                                pagination={'rowsPerPage': 20}
                            ).classes('w-full')
                            
                            # Добавляем форматирование
                            tk_group_table.add_slot('body-cell-group', '''
                                <q-td key="group" :props="props" style="text-align: left;">
                                    <div class="text-bold">
                                        {{ props.value }}
                                    </div>
                                </q-td>
                            ''')
                            
                            tk_group_table.add_slot('body-cell-forecast', '''
                                <q-td key="forecast" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 0}).format(props.value) }}
                                    </div>
                                </q-td>
                            ''')
                            
                            tk_group_table.add_slot('body-cell-percentage', '''
                                <q-td key="percentage" :props="props" style="text-align: right;">
                                    <div class="font-mono text-bold">
                                        {{ props.value.toFixed(1) }}%
                                    </div>
                                </q-td>
                            ''')
                            
                            tk_group_table.rows = rows
                    
        await self.update_progress(100, 'Готово!')
        await asyncio.sleep(2)
        progress_container.clear()
    
    def on_load_all_click(self):
        """Загрузка всех данных"""
        ui.notify('Загрузка данных...', type='info')
        self.update_years_table()
        self.df_sales = self.load_sales_data()
        self.df_directions = self.load_directions_data()
        
        if not self.df_sales.empty:
            self.monthly_df = self.prepare_monthly_data(self.df_sales)
            self.weekly_df = self.prepare_weekly_data(self.df_sales)
            self.daily_df = self.prepare_daily_data(self.df_sales)
            ui.notify(f'Загружено {len(self.df_sales)} записей о продажах', type='positive')
        
        if not self.df_directions.empty:
            ui.notify(f'Загружено {len(self.df_directions)} записей по направлениям', type='positive')
        
        # Очищаем прогнозы
        forecast_container.clear()
        direction_container.clear()
    
    def create_ui(self):
        """Создание интерфейса"""
        # Добавляем CSS для единого стиля таблиц
        ui.add_head_html('''
        <style>
        .font-mono {
            font-family: 'Courier New', monospace !important;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        .q-table td, .q-table th {
            padding: 12px 16px !important;
            font-size: 14px;
        }
        .q-table th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .q-table tbody tr:hover {
            background-color: #f9f9f9;
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
                [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027],
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
        
        # Контейнер для общего прогноза
        global forecast_container
        forecast_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4 mb-8')
        
        # Контейнер для прогноза по направлениям
        global direction_container
        direction_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

        # Контейнер для прогноза по товарным группам ОАИ
        global oai_group_container
        oai_group_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

        # Контейнер для прогноза по товарным группам ТК
        global tk_group_container
        tk_group_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

        # Контейнер для прогноза по товарным группам РЕКЛАМА
        global reklama_group_container
        reklama_group_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

        # Контейнер для прогноза по товарным группам КН
        global kn_group_container
        kn_group_container = ui.column().classes('w-full max-w-4xl mx-auto mt-4')

# Запуск приложения
app = ForecastApp()

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=True)