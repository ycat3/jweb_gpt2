use rust_bert::gpt2::GPT2Generator;
use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{ RemoteResource,  ResourceProvider};
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model_resource = Box::new(RemoteResource {     
        url: "https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/rust_model.ot".into(),
        cache_subdir: "japanese-soseki-gpt2-1b/model".into(),        
    });
    let config_resource = Box::new(RemoteResource {     
        url: "https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/config.json".into(),
        cache_subdir: "japanese-soseki-gpt2-1b/config".into(),        
    });
    let vocab_resource = Box::new(RemoteResource {     
        url: "https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/spiece.model".into(),
        cache_subdir: "japanese-soseki-gpt2-1b/vocab".into(),        
    });
    let vocab_resource_token = vocab_resource.clone();
    let merges_resource = vocab_resource.clone();    
    let generate_config = GenerateConfig {        
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource, // not used        
        device: Device::Cpu,
        repetition_penalty: 1.6,
        min_length: 40,
        max_length: 128,
        do_sample: true,
        early_stopping: true,
        num_beams: 5,
        temperature: 1.0,
        top_k: 500,
        top_p: 0.95,
        ..Default::default()
    };
    let tokenizer = TokenizerOption::from_file(
        ModelType::T5,
        vocab_resource_token.get_local_path().unwrap().to_str().unwrap(),
        None,
        true,
        None,
        None,
    )?;
    let mut gpt2_model = GPT2Generator::new_with_tokenizer(generate_config, tokenizer.into())?;
    gpt2_model.set_device(Device::cuda_if_available());
    let input_text = "夏目漱石は、";
    let t1 = std::time::Instant::now();
    let output = gpt2_model.generate(Some(&[input_text]), None);
    println!("{}", output[0].text);
    println!("Elapsed Time(ms):{}",t1.elapsed().as_millis()); 
    Ok(())
}
// sample output: 夏目漱石は、明治から大正にかけて活躍した日本の小説家です。彼は「吾輩は猫である」や「坊っちゃん」、「草枕」「三四郎」、あるいは「虞美人草」などの小説で知られていますが、「明暗」のような小説も書いていました。
