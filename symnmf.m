function symnmf(path,tar_rank)
	tar_rank=str2num(tar_rank);
	addpath(path);
	data = load("term_corelation.mat");
	data= cat(1,cellfun(@(x) data.(x),sort(fieldnames(data)),'uni',0){:});
	[U,iter,obj] = symnmf_anls(data,tar_rank);
	save(strcat("term_topic_",num2str(tar_rank)),'U',"-6");
	disp("Symmetric nmf done, Calculating reconstruction error")
	norm_diff = norm(data-(U*U'));
	fid = fopen(strcat(strcat("term_topic_details_",num2str(tar_rank)),".txt"),'w');
	fprintf(fid,"Reconstruction error: %f\n",norm_diff);
	fprintf(fid,"No. of iterations: %f\n",iter)
	fprintf(fid,"Value of obj: %f\n",obj)
	fclose(fid);