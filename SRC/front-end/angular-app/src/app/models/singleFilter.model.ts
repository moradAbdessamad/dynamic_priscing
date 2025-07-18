const today = new Date();

const formatDate = (date: Date): string => {
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  return `${year}-${month}-${day}`;
};

export interface SingleDateFilterModel {
  singleDate: string;
  unit: number;
  targetPredict: string;
  models: string;
  targetValueMax: number;
  targetValueMin: number;
}

export const defaultSingleDateFilters: SingleDateFilterModel = {
  singleDate: formatDate(today),
  unit: 166,
  targetPredict: 'price',
  models: 'LinearRegression',
  targetValueMax: 90,
  targetValueMin: 10
};

export { unitsOptions, modelsOptions, targetPredictOptions } from './filter.model';