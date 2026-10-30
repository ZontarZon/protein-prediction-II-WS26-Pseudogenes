export type DatasetGroup = { key: string; label: string; count: number; subset_parameter?: string };
export function selectionEstimate(groups: DatasetGroup[], parameters: Record<string, unknown>, validationParameter?: string) {
  const batch = Number(parameters.batch_size ?? 1);
  const percent = validationParameter ? Number(parameters[validationParameter] ?? 0) : 0;
  const valid = Number.isInteger(batch) && batch > 0 && Number.isFinite(percent) && percent >= 0 && percent < 100 && groups.every((group) => {
    const count = Number(group.subset_parameter ? parameters[group.subset_parameter] ?? 0 : 0);
    return Number.isInteger(count) && count >= 0;
  });
  const selected = groups.map((group) => {
    const limit = Number(group.subset_parameter ? parameters[group.subset_parameter] ?? 0 : 0);
    const count = limit > 0 ? Math.min(limit, group.count) : group.count;
    const validation = percent > 0 && count >= 2 ? Math.max(1, Math.min(count - 1, Math.floor(count * percent / 100 + .5))) : 0;
    return { ...group, selected: count, validation, training: count - validation };
  });
  const training = selected.reduce((sum, group) => sum + group.training, 0);
  const validation = selected.reduce((sum, group) => sum + group.validation, 0);
  return { valid, groups: selected, total: training + validation, training, validation,
    trainingBatches: valid ? Math.ceil(training / batch) : 0,
    validationBatches: valid ? Math.ceil(validation / batch) : 0 };
}
