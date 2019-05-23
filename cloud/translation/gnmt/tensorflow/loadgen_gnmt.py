from generic_loadgen import *
import sys
from nmt.nmt import create_hparams, add_arguments, create_or_load_hparams
import argparse
from nmt.inference import get_model_creator, start_sess_and_load_model, load_data
import tensorflow as tf
import os
from nmt.utils import misc_utils as utils
from nmt.utils import nmt_utils
from nmt import model_helper
import codecs

class TranslationTask:
    def __init__(self, query_id, sentence_id, output_file):
        self.query_id = query_id
        self.sentence_id = sentence_id
        self.output_file = output_file
        self.start = time.time()

class BatchTranslationTask:
    def __init__(self, sentence_id_list, query_id_list):
        self.sentence_id_list = sentence_id_list
        self.query_id_list = query_id_list
        self.query_id = self.query_id_list[0]   #FIXME generic_loadgen needs this

##
# @brief Wrapper around TF GNMT Inference that can interface with loadgen
class GNMTRunner (Runner):
    ##
    # @brief Constructor will build the graph and set some wrapper variables
    # @param input_file: path to the input text
    # @param ckpt_path: path to the GNMT checkpoint
    # @param hparams_path: path to the parameters used to configure GNMT graph
    # @param vocab_prefix: Path to vocabulary file (note: don't add .en or .de suffixes)
    # @param outdir: Output directory to optionally write translations to
    # @param batch_size: batch size to use when processing BatchTranslationTasks
    def __init__(self, input_file=None, ckpt_path=None, hparams_path=None, vocab_prefix=None, outdir=None, batch_size=32):
        Runner.__init__(self)

        # If no value is provided for the construtor arguments, set defaults here
        if input_file is None:
            input_file = os.path.join(os.getcwd(), 'nmt', 'data', 'newstest2014.tok.bpe.32000.en')

        if ckpt_path is None:
            ckpt_path = os.path.join(os.getcwd(), 'ende_gnmt_model_4_layer',
                        'translate.ckpt')

        if hparams_path is None:
            hparams_path= os.path.join(os.getcwd(), 'nmt', 'standard_hparams',
                             'wmt16_gnmt_4_layer.json')

        if vocab_prefix is None:
            vocab_prefix = os.path.join(os.getcwd(), 'nmt', 'data', 'vocab.bpe.32000')

        if outdir is None:
            outdir = os.path.join(os.getcwd(), 'lg_output')

        flags = self.parse_options(ckpt_path, hparams_path, vocab_prefix, outdir, batch_size)

        self.setup(flags)

        # Wrapper parameters
        self.input_file = input_file
        self.infer_data = []  # This will be filled by load_samples_to_ram
        self.count = 0

    ##
    # @brief Parse GNMT-specific options before setting up
    def parse_options(self, ckpt_path, hparams_path, vocab_prefix, outdir, batch_size):
        FLAGS = None
        # TBD remove argument parsing, and just have it return all default values.
        nmt_parser = argparse.ArgumentParser()
        add_arguments(nmt_parser)
        FLAGS, unparsed = nmt_parser.parse_known_args()

        # Some of these flags are never used and are just set for consistency
        FLAGS.num_workers = 1
        FLAGS.iterations = 1
        FLAGS.infer_batch_size = batch_size
        FLAGS.num_inter_threads = 1
        FLAGS.num_intra_threads = 1
        FLAGS.run = "accuracy" # Needs to be set to accuracy to generate output

        # Pass in inference specific flags
        FLAGS.ckpt = ckpt_path
        FLAGS.src = 'en'
        FLAGS.tgt = 'de'
        FLAGS.hparams_path = hparams_path
        FLAGS.out_dir = outdir
        FLAGS.vocab_prefix = vocab_prefix
        
        return FLAGS

    ##
    # @brief Configure hparams and setup GNMT graph 
    # @pre Requires output from parse_options
    def setup(self, flags):
        # Model output directory
        out_dir = flags.out_dir
        if out_dir and not tf.gfile.Exists(out_dir):
          tf.gfile.MakeDirs(out_dir)

        # Load hparams.
        default_hparams = create_hparams(flags)
        loaded_hparams = False
        if flags.ckpt:  # Try to load hparams from the same directory as ckpt
          ckpt_dir = os.path.dirname(flags.ckpt)
          ckpt_hparams_file = os.path.join(ckpt_dir, "hparams")
          if tf.gfile.Exists(ckpt_hparams_file) or flags.hparams_path:
                # Note: for some reason this will create an empty "best_bleu" directory and copy vocab files
                hparams = create_or_load_hparams(ckpt_dir, default_hparams, flags.hparams_path, save_hparams=False)
                loaded_hparams = True
        
        assert loaded_hparams

        # GPU device
        config_proto = utils.get_config_proto(
            allow_soft_placement=True,
            num_intra_threads=hparams.num_intra_threads,
            num_inter_threads=hparams.num_inter_threads)
        utils.print_out(
            "# Devices visible to TensorFlow: %s" 
            % repr(tf.Session(config=config_proto).list_devices()))


        # Inference indices (inference_indices is broken, but without setting it to None we'll crash)
        hparams.inference_indices = None
        
        # Create the graph
        model_creator = get_model_creator(hparams)
        infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)
        sess, loaded_infer_model = start_sess_and_load_model(infer_model, flags.ckpt,
                                                       hparams)

        # Parameters needed by TF GNMT
        self.hparams = hparams
        self.out_dir = out_dir

        self.infer_model = infer_model
        self.sess = sess
        self.loaded_infer_model = loaded_infer_model

    ##
    # @brief Load sentences into the infer_data array
    def load_samples_to_ram(self, query_samples):
        self.infer_data = load_data(self.input_file, self.hparams)


    ##
    # @brief Run translation on a number of sentence id's
    # @param sentence_id_list: List of sentence numbers to translate
    # @return Translated sentences
    def translate(self, sentence_id_list):
        infer_mode = self.hparams.infer_mode

        # Set input data and batch size
        with self.infer_model.graph.as_default():
            self.sess.run(
                self.infer_model.iterator.initializer,
                feed_dict={
                    self.infer_model.src_placeholder: [self.infer_data[i] for i in sentence_id_list],
                    self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size
                })

        # Start the translation
        nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
        if infer_mode != "beam_search":
          nmt_outputs = np.expand_dims(nmt_outputs, 0)

        batch_size = nmt_outputs.shape[1]
        assert batch_size <= self.hparams.infer_batch_size

        # Whether beam search is being used or not, we only want 1 final translation
        assert self.hparams.num_translations_per_input == 1

        translation = []
        for decoded_id in range(batch_size):
            translation += [nmt_utils.get_translation(
                        nmt_outputs[0],
                       decoded_id,
                       tgt_eos=self.hparams.eos,
                       subword_option=self.hparams.subword_option)]

        return translation

    ##
    # @brief Invoke GNMT to translate the input file
    # @pre Ensure load_samples_to_ram was called to fill self.infer_data
    def process(self, qitem):
        translation = self.translate(qitem.sentence_id_list)

        # Keeping track of how many translations happened
        self.count += len(translation)

        print("Performed {} translations".format(self.count))
        
        return translation

    ##
    # @brief Create a batched task and add it to the queue
    def enqueue(self, query_samples):
        bs = self.hparams.infer_batch_size
        num_samples = len(query_samples)
        for i in range(0, num_samples, bs):
            query_id_list = [sample.id for sample in query_samples[i:min(i+bs, num_samples)]]
            sentence_id_list = [sample.index for sample in query_samples[i:min(i+bs, num_samples)]] 
            task = BatchTranslationTask(sentence_id_list, query_id_list)
            self.tasks.put(task)

##
# @brief Subclass of GNMTRunner, specialized for batch size 1
class SingleStreamGNMTRunner (GNMTRunner):
    ##
    # @brief Constructor will build the graph and set some wrapper variables
    # @param input_file: path to the input text
    # @param ckpt_path: path to the GNMT checkpoint
    # @param hparams_path: path to the parameters used to configure GNMT graph
    # @param vocab_prefix: Path to vocabulary file (note: don't add .en or .de suffixes)
    # @param outdir: Output directory to optionally write translations to
    # @param store_translation: whether output should be stored
    def __init__(self, input_file=None, ckpt_path=None, hparams_path=None, vocab_prefix=None, outdir=None, store_translation=False):
        GNMTRunner.__init__(self, input_file, ckpt_path, hparams_path, vocab_prefix, outdir, batch_size=1)

        self.store_translation = store_translation


    ##
    # @brief Invoke GNMT to translate the input file
    # @pre Ensure load_samples_to_ram was called to fill self.infer_data
    def process(self, qitem):
        if self.store_translation:
            print("translate {} (QID {}): Sentence ID {} --> {}".format(self.count, qitem.query_id, qitem.sentence_id, qitem.output_file))
       
        sentence_id = qitem.sentence_id 

        translation = self.translate([sentence_id])

        if self.store_translation:
            assert len(translation) == 1
            self.write_output(translation[0], qitem.output_file)

        # Keeping track of how many translations happened
        self.count += 1
        
        return translation

    ##
    # @brief Write translation to file
    def write_output(self, translation, trans_file):
          with codecs.getwriter("utf-8")(
              tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write((translation + b"\n").decode("utf-8"))

    ##
    # @brief Create a task and add it to the queue
    def enqueue(self, query_samples):
        for sample in query_samples:
            sentence_id = sample.index
            output_file = os.path.join(self.out_dir, "sentence_{}_de".format(sample.index))

            task = TranslationTask(sample.id, sentence_id, output_file)
            self.tasks.put(task)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default='SingleStream',
                            help="Scenario to be run: can be one of {SingleStream, Offline}")


    parser.add_argument('--store_translation', default=False, action='store_true',
                            help="Store the output of translation? Note: Only valid with SingleStream scenario.")

    args = parser.parse_args()

    settings = mlperf_loadgen.TestSettings()
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly

    if args.scenario == "SingleStream":
        runner = SingleStreamGNMTRunner(store_translation=args.store_translation)

        settings.scenario = mlperf_loadgen.TestScenario.SingleStream
        
        # Specify exactly how many queries need to be made
        settings.enable_spec_overrides = True
        settings.override_min_query_count = 3003
        settings.override_max_query_count = 3003

    elif args.scenario == "Offline":
        runner = GNMTRunner(batch_size=32)

        settings.scenario = mlperf_loadgen.TestScenario.Offline
        
        # Specify exactly how many queries need to be made
        settings.enable_spec_overrides = True
        settings.override_min_query_count = 1
        settings.override_max_query_count = 1

    else:
        print("Invalid scenario selected")
        assert False

    # Create a thread in the GNMTRunner to start accepting work
    runner.start_worker()

    total_queries = 3003 # Maximum sample ID + 1
    perf_queries = 3003   # Select the same subset of $perf_queries samples

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, runner.load_samples_to_ram, runner.unload_samples_from_ram)

    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

