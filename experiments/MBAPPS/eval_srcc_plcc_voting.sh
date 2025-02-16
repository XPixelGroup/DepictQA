src_dir=../../src/
data_dir=../../../DataDepictQA/

python $src_dir/eval/cal_srcc_plcc_voting.py \
    --test_paths $data_dir/KADID10K/metas_srcc_plcc_voting/test_refAB_single_kadid_124k.json \
                 $data_dir/CSIQ/metas_srcc_plcc_voting/test_refAB_single_csiq_12k.json \
                 $data_dir/TID2013/metas_srcc_plcc_voting/test_refAB_single_tid2013_179k.json \
    --pred_paths answers/quality_compare_kadid_srcc_plcc_voting.json \
                 answers/quality_compare_csiq_srcc_plcc_voting.json \
                 answers/quality_compare_tid_srcc_plcc_voting.json \
    --gt_paths  $data_dir/KADID10K/metas_srcc_plcc_voting/test_ref_dist_mos_kadid.json \
                $data_dir/CSIQ/metas_srcc_plcc_voting/test_ref_dist_mos_csiq.json \
                $data_dir/TID2013/metas_srcc_plcc_voting/test_ref_dist_mos_tid2013.json \
    --confidence \
