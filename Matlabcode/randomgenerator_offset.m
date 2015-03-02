function  [u1,u2] = randomgenerator_offset(low_bound,high_bound)


u1 = round(low_bound + (high_bound - low_bound)*rand(1));
if rand(1)>0.5;
    u1 = -u1;
end
u2 = round(low_bound + (high_bound - low_bound)*rand(1));
if rand(1)>0.5;
    u2 = -u2;
end

end