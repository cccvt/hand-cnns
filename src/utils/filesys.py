import os


def mkdir(path, safeguard=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if safeguard:
            print('\033[93m folder {} already exists! \033[0m'.format(path))
            answer = input('Do you want to continue [y|n]?')
            if answer[0].lower() == 'n':
                raise SystemExit('Exiting according to user instruction')
            else:
                print('Continuing with existing directory')
