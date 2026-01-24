function data = LoadData(sel)
    rootpath = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    if sel == "train_test"
        % train_x, train_y, test_x, test_y
        data = load(fullfile(rootpath, 'data/pan/train_test_data.mat'));

    elseif sel == "train_val_test"
        % train_x, train_y, val_x, val_y, test_x, test_y
        data = load(fullfile(rootpath, 'data/pan/train_val_test_data.mat'));

    elseif sel == "train1_train2_val_test"
        % train_x1, train_y1, train_x2, train_y2, val_x, val_y, test_x, test_y
        data = load(fullfile(rootpath, 'data/pan/train1_train2_val_test_data.mat'));
    
    else
        fprintf('输入sel不合法！%s \n', sel)
    end
end