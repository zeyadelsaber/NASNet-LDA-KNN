function paths = project_paths(requireDataset)
%PROJECT_PATHS Return repository, dataset, figure, and result locations.
% Set TARTANAIR_DATASET_ROOT when the dataset is stored outside this repo.

arguments
    requireDataset (1,1) logical = true
end

repoRoot = fileparts(mfilename("fullpath"));
datasetRoot = string(getenv("TARTANAIR_DATASET_ROOT"));

if strlength(datasetRoot) == 0
    datasetRoot = fullfile(repoRoot, "data");
end

paths.repo = string(repoRoot);
paths.dataset = datasetRoot;
paths.training = fullfile(datasetRoot, "Training");
paths.validation = fullfile(datasetRoot, "Validation");
paths.test = fullfile(datasetRoot, "Test");
paths.figures = fullfile(repoRoot, "figures");
paths.results = fullfile(repoRoot, "results");
paths.models = fullfile(paths.results, "models");

requiredFolders = [paths.training, paths.validation, paths.test];
if requireDataset && ~all(isfolder(requiredFolders))
    error("TartanAir:DatasetNotFound", ...
        "Dataset folders were not found under '%s'. " + ...
        "Create data/Training, data/Validation, and data/Test, or set " + ...
        "TARTANAIR_DATASET_ROOT.", datasetRoot);
end

if ~isfolder(paths.figures), mkdir(paths.figures); end
if ~isfolder(paths.results), mkdir(paths.results); end
if ~isfolder(paths.models), mkdir(paths.models); end
end
