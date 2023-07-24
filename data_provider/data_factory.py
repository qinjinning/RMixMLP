from data_provider.data_loader import Dataset_CMPASS,Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'FD001': Dataset_CMPASS,
    'FD002': Dataset_CMPASS,
    'FD003': Dataset_CMPASS,
    'FD004': Dataset_CMPASS,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        # 
        # if args.data == 'FD002':
        #     batch_size = 256
        # if args.data == 'FD004':
        #     batch_size = 248
    
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred  
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        train_path=args.train_path,
        test_path=args.test_path,
        rul_path=args.rul_path,
        seq_len=args.seq_len,
        engine_num = args.engine_num
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader