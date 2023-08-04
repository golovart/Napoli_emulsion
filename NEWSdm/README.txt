make a bash-script to produce images.
remember the order:
> create csvs/
> copy get_ims.C
> root dm_tracks_cl.dm.root
> gSystem->Load("libDMRoot")
> .x get_ims.C(files, dr)
