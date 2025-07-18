const today = new Date();
const fiveDaysFromToday = new Date();
fiveDaysFromToday.setDate(today.getDate() + 5);

const formatDate = (date: Date): string => {
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  return `${year}-${month}-${day}`;
};

export interface FilterModel {
  fromDate: string;
  toDate: string;
  unit: number;
  targetPredict: string;
  models: string;
  targetValueMax: number;
  targetValueMin: number;
}

export const defaultFilters: FilterModel = {
  fromDate: formatDate(today),
  toDate: formatDate(fiveDaysFromToday),
  unit: 166,
  targetPredict: 'price',
  models: 'LinearRegression',
  targetValueMax: 90,
  targetValueMin: 10
};

export const unitsOptions = [
  { value: 166, label: 'Standard Room' },
  { value: 167, label: 'Family Room' }
];

export const modelsOptions = [
  { value: 'XGBoost', label: 'XGBoost' },
  { value: 'LSTM', label: 'LSTM' },
  { value: 'LinearRegression', label: 'Linear Regression' }
];

export const targetPredictOptions = [
  { value: 'price', label: 'Price' },
  { value: 'taux', label: 'Taux' }
];