def eval(*,
         student='student.th', 
         classifier='../teachers/clf-256-resnet/clf.th', 
         dataroot='/home/mcherti/work/data/insects/train_img_classes', 
         batchSize=32, 
         imageSize=256,
         nb_classes=18):

    sys.path.append(os.path.dirname(classifier))
    sys.path.append(os.path.dirname(classifier) + '/..')
 
    clf = torch.load(classifier)

    if 'cifar10' in dataroot:
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    clf_mean = Variable(torch.FloatTensor(mean).view(1, -1, 1, 1)).cuda()
    clf_std = Variable(torch.FloatTensor(std).view(1, -1, 1, 1)).cuda()

    S = torch.load(student)
    S.train(False)
    input = torch.zeros(batchSize, 3, imageSize, imageSize)
    input = Variable(input)
    input = input.cuda()
    
    if 'cifar10' in dataroot:
        transform = transforms.Compose([
           transforms.Scale(imageSize),
           transforms.CenterCrop(imageSize),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform, train=True)

    else:
        transform = transforms.Compose([
               transforms.Scale(imageSize),
               transforms.CenterCrop(imageSize),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImageFolder(root=dataroot, transform=transform)


    torch.cuda.manual_seed(42)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchSize,
        sampler=RandomSampler(dataset),
        num_workers=8)
    accs_student = []
    accs_teacher = []
    accs_student_teacher = []
    for b, (X, y) in enumerate(dataloader):
        t = time.time()
        batch_size = X.size(0)
        input.data.resize_(X.size()).copy_(X)
        y_true = torch.zeros(batch_size, nb_classes)
        y_true.scatter_(1, y.view(y.size(0), 1), 1)
        y_teacher = clf(norm(input, clf_mean, clf_std))
        y_student = S(input)
        acc_teacher = get_acc(y_true, y_teacher.data.cpu())
        acc_student = get_acc(y_true, y_student.data.cpu())
        acc_student_teacher = get_acc(y_teacher.data.cpu(), y_student.data.cpu())
        accs_student.append(acc_student)
        accs_teacher.append(acc_teacher)
        accs_student_teacher.append(acc_student_teacher)
        print('acc student : {:.3f}, acc teacher : {:.3f}, acc student on teacher : {:.3f}'.format(np.mean(accs_student), np.mean(accs_teacher), np.mean(accs_student_teacher)))
        dt = time.time() - t

