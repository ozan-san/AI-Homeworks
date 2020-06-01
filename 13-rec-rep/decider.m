function decision = decider(ground_truth, selected_classifier, new_classifier, priority)
% we will assume a function computing getConfusionMatrix
% this is also presented in the .zip file as "getConfusionMatrix.m"
% Precision = TP / (TP+FP)
% Recall    = TP / (TP+FN)
% F_beta = (1 + Beta^2) * (Precision * Recall)
%                        ----------------------
%                      (Beta^2 * Precision) + Recall


% keep in mind that we have selected C4 as the best for our specific
% use case. We can assign selected_classifier = C4 if need be.

baseline_fscore = 0;
new_fscore = 0;
if priority == "general"
   beta = 1.0; 
elseif priority == "precision"
   beta = 0.5;
elseif priority == "recall"
   beta = 2.0;
end
    

for i=1:size(new_classifier,2)
    conf_new = getConfusionMatrix(GT, new_classifier(:, i));
    precision = conf_new.tp / (conf_new.tp + conf_new.fp);
    recall = conf_new.tp / (conf_new.tp + conf_new.fn);
    fbeta = (1 + beta*beta) * (precision * recall) / (beta*beta*precision) + recall;
    new_fscore = new_fscore + fbeta;
end

for i=1:size(selected_classifier, 2)
    conf_baseline = getConfusionMatrix(GT, selected_classifier(:, i));
    precision = conf_baseline.tp / (conf_baseline.tp + conf_baseline.fp);
    recall = conf_baseline.tp / (conf_baseline.tp + conf_baseline.fn);
    fbeta = (1 + beta*beta) * (precision * recall) / (beta*beta*precision) + recall;
    baseline_fscore = baseline_fscore + fbeta; 
end

new_fscore = new_fscore / size(new_classifier, 2);
baseline_fscore = baseline_fscore / size(selected_classifier, 2);

decision = (new_fscore > baseline_fscore);
end
