        from keras.utils import multi_gpu_model

        parallel_model = multi_gpu_model(model,gpus=2)
        parallel_model.compile(loss='binary_crossentropy', optimizer=opts.Adam(lr=1e-4, decay=1e-6), metrics=['acc'])
        #### temp put it here, to be put it another place
        class ParallelModelCheckpoint(ModelCheckpoint):
            def __init__(self,model,filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,mode='auto', period=1):
                self.single_model = model
                super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)
            def set_model(self, model):
                super(ParallelModelCheckpoint,self).set_model(self.single_model)
        #checkpoint = ParallelModelCheckpoint(model, filepath='./cifar10_resnet_ckpt.h5', monitor='val_acc', verbose=1, save_best_only=True) # 解决多GPU运行下
保存模型报错的问题


        mc = ParallelModelCheckpoint(model,filepath='../user_data/h5s/{}_best_parallel.h5'.format(args.model_name),verbose=1,save_best_only=True)
        rlop = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=1,verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
        #history = model.fit(merge_feats_train,merge_labels_train, epochs=args.epochs,validation_data=(merge_feats_val,merge_labels_val), callbacks=[mc,rlop, es], verbose=1)
        history = parallel_model.fit(merge_feats_train,merge_labels_train, epochs=args.epochs,batch_size=args.batch_size,validation_data=(merge_feats_val,merge_labels_val), callbacks=[mc,rlop, es], verbose=1)
        val_acc = history.history['val_acc']
        np.save('../user_data/npys/{}_val_acc.npy'.format(args.model_name),val_acc)

