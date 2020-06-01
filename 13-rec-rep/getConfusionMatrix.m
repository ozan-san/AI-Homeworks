function confusionMatrix=getConfusionMatrix(ground_truth, labels)
s = size(ground_truth, 1);
fp = 0;
fn = 0;
tp = 0;
tn = 0;
for i=1:s
    if ground_truth(i) == labels(i)
        % true... something
        if ground_truth(i) == 1
            tp = tp + 1;
        else
            tn = tn + 1;
        end
    else
        if ground_truth(i) == 0
            fp = fp + 1;
        else
            fn = fn + 1;
        end
    end

end
confusionMatrix.fp = fp;
confusionMatrix.fn = fn;
confusionMatrix.tp = tp;
confusionMatrix.tn = tn;
end