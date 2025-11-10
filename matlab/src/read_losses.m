function [l_0_matrix, l_1_matrix] = read_losses(filename)

% [l_0_matrix, l_1_matrix] = read_lv(filename)
% 
% Reads the files and stores the loss-based metrics in matrices

% Read the file
fileContent = fileread(filename);

% Define labels to extract
labels = {'l_0', 'l_1'};

% Initialize storage for extracted values
data = struct();

for i = 1:length(labels)
    label = labels{i};

    % Regular expression to find arrays inside parentheses
    pattern = strcat(label, '\s*:\s*\((.*?)\)');
    matches = regexp(fileContent, pattern, 'tokens');

    % Convert matches into numeric arrays
    extractedArrays = cellfun(@(x) str2num(x{1}), matches, 'UniformOutput', false); %#ok<ST2NM>

    % Store in struct
    data.(label) = extractedArrays;
end

% Convert cell arrays to matrices if all vectors have the same length
fields = fieldnames(data);
for i = 1:length(fields)
    fieldName = fields{i};
    cellArray = data.(fieldName);

    % If all arrays have the same length, convert to a matrix
    if all(cellfun(@length, cellArray) == length(cellArray{1}))
        data.(fieldName) = cell2mat(cellArray'); % Convert to matrix
    end
end

% Access extracted data
l_0_matrix = data.l_0; % Matrix if possible, otherwise cell array
l_1_matrix = data.l_1;
