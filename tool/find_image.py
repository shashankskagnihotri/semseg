

cospgd = "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/cospgd/alpha_015/voc2012/unet_11/high_alpha/new_corrent_cospgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/log.log"
segpgd= "/work/ws-tmp/sa058646-segment/semseg/runs/final_icml/segpgd/voc2012/unet_11/correct_segpgd_attack/convnext_tiny_trans_kernel_2_small_trans_0_convnext_backbone_False_backbone_kernel_3_small_conv_0/val/ss/iterations_20/log.log"

with open(cospgd) as textfile1, open(segpgd) as textfile2: 
        for x, y in zip(textfile1, textfile2):
            x = x.strip()
            y = y.strip()
            if '1449 on image' in x:
                image_name = x.split("image ",1)[1].split(',',1)[0]
                #print('image name: ', image_name)
                cos_num = x.split("accuracy ",1)[1].split('.',2)[1]                
                cos_num=float('0.'+cos_num)
                seg_num = y.split("accuracy ",1)[1].split('.',2)[1]
                seg_num=float('0.'+seg_num)
                if seg_num > cos_num:
                    print("image: {}\t cos: {}\t seg: {}".format(image_name, cos_num, seg_num))
                #print(f"{x}\t{y}")