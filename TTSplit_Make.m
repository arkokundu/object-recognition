function TTS = TTSplit_Make(AvailIndexList,TrainPart, TestPart)
%
%
nfiles = length(AvailIndexList);
if((TrainPart + TestPart) > 1)
  error('TrainPart + TestPart > 1\n');
  TTS = [];
  return;
end
p = randperm(nfiles);
nTrain = floor(TrainPart * nfiles);
nTest = floor(TestPart * nfiles);
TTS.Train = p(1:nTrain);
TTS.Train = AvailIndexList(TTS.Train);
TTS.Test = p((nTrain+1):(nTrain+nTest));
TTS.Test = AvailIndexList(TTS.Test);

