import {
	OpenAI,
	SimpleDirectoryReader,
	ChatMessage,
	ChatResponse,
	Document,
	VectorStoreIndex,
	QueryEngine,
	RetrieverQueryEngine,
	MessageContentDetail,
	Metadata,
	PapaCSVReader,
	DocxReader,
	HTMLReader,
	MarkdownReader,
	TextFileReader,
	PDFReader,
} from 'llamaindex';
import fs from 'node:fs';
import utils from 'node:util';
import path from 'node:path';

import { config } from './config';

interface Completion {
	Content: string | null;
	TokenUsage: number | undefined;
}

interface ConnectorResponse {
	Completions: Completion[];
	ModelType: string;
}

interface RawResponse {
	model?: string;
	usage?: {
		prompt_tokens?: number;
		completion_tokens?: number;
		total_tokens?: number;
	};
}

interface Message extends ChatResponse {
	raw: RawResponse | null;
}

interface ErrorCompletion {
	choices: Array<{
		message: {
			content: string;
		};
	}>;
	error: string;
	model: string;
	usage: undefined;
}

interface DocumentWithIndex {
	indexFromDocument: (QueryEngine & RetrieverQueryEngine) | undefined;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const logger = (logBody: any, description?: string): void => {
	if (description) console.log(`${description}`);

	console.log(
		utils.inspect(logBody, {
			showHidden: false,
			depth: null,
			colors: true,
		}),
	);
};

const mapToResponse = (
	outputs: (Message | ErrorCompletion)[],
	model: string,
): ConnectorResponse => {
	return {
		Completions: outputs.map((output: Message) => {
			if ('error' in output) {
				return {
					Content: null,
					TokenUsage: undefined,
					Error: output.error,
				};
			}

			return {
				Content: output.message.content as string,
				TokenUsage: output.raw?.usage?.prompt_tokens,
			};
		}),
		ModelType: model,
	};
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mapErrorToCompletion = (error: any, model: string): ErrorCompletion => {
	const errorMessage = error.message || JSON.stringify(error);
	return {
		choices: [],
		error: errorMessage,
		model,
		usage: undefined,
	};
};

async function checkDirectory(path: string): Promise<boolean | undefined> {
	try {
		const stat = await fs.promises.stat(path);
		return stat.isDirectory();
	} catch (error) {
		logger(error);
	}
}

enum Extension {
	pdf = '.pdf',
	csv = '.csv',
	docx = '.docx',
	html = '.html',
	md = '.md',
	txt = '.txt',
}
const mapReader = {
	[Extension.pdf]: new PDFReader(),
	[Extension.csv]: new PapaCSVReader(),
	[Extension.docx]: new DocxReader(),
	[Extension.html]: new HTMLReader(),
	[Extension.md]: new MarkdownReader(),
	[Extension.txt]: new TextFileReader(),
};

async function workWithExtensions(
	url: string,
	extension: Extension,
): Promise<DocumentWithIndex | undefined> {
	const reader = mapReader[extension];

	const documents = await reader.loadData(url);

	const indexDB = await VectorStoreIndex.fromDocuments(documents);

	return {
		indexFromDocument: indexDB.asQueryEngine() as unknown as QueryEngine &
			RetrieverQueryEngine,
	};
}

async function workWithFolder(
	directoryPath: string,
): Promise<DocumentWithIndex[] | undefined> {
	try {
		const directoryReader = new SimpleDirectoryReader();
		const documents = await directoryReader.loadData(directoryPath);

		return Promise.all(
			documents.map(async (doc) => {
				const document = new Document({ text: doc.text });
				const index = await indexFromDocument(document);

				return {
					text: doc.text,
					indexFromDocument: index,
				};
			}),
		);
	} catch (error) {
		logger(error);
	}
}

async function readFile(url: string): Promise<string | undefined> {
	try {
		return await fs.promises.readFile(url, 'utf-8');
	} catch (error) {
		logger(error, `Can't read ${url}`);
	}
}

const mapFileByExtensionToParser: {
	[key in Extension]: (path: string) => Promise<DocumentWithIndex | undefined>;
} = {
	[Extension.pdf]: (path) => workWithExtensions(path, Extension.pdf),
	[Extension.csv]: (path) => workWithExtensions(path, Extension.csv),
	[Extension.docx]: (path) => workWithExtensions(path, Extension.docx),
	[Extension.html]: (path) => workWithExtensions(path, Extension.html),
	[Extension.md]: (path) => workWithExtensions(path, Extension.md),
	[Extension.txt]: (path) => workWithExtensions(path, Extension.txt),
};

async function workWithFile(
	filePath: string,
): Promise<DocumentWithIndex | undefined> {
	try {
		const ext = path.extname(filePath);

		const unknownExtension = async (
			path: string,
		): Promise<DocumentWithIndex | undefined> => {
			const text = (await readFile(path)) as string;
			const document = new Document({ text });
			const indexDB = await VectorStoreIndex.fromDocuments([document]);
			const index = indexDB.asQueryEngine() as unknown as QueryEngine &
				RetrieverQueryEngine;

			return {
				indexFromDocument: index as unknown as QueryEngine &
					RetrieverQueryEngine,
			};
		};

		const supportExtensions = Object.values(Extension) as string[];

		return supportExtensions.includes(ext)
			? mapFileByExtensionToParser[ext as Extension](filePath)
			: unknownExtension(filePath);
	} catch (error) {
		logger(error, 'workWithFile function');
	}
}

async function parseDocument(
	path: string,
): Promise<
	Promise<DocumentWithIndex[]> | Promise<DocumentWithIndex> | undefined
> {
	try {
		const isADirectory = await checkDirectory(path);
		if (isADirectory) return await workWithFolder(path);
		else return await workWithFile(path);
	} catch (error) {
		logger(error, 'Parse document');
	}
}

function extractDocumentUrls(prompt: string): string[] {
	const urlRegex =
		/(https?:\/\/[^\s]+|[a-zA-Z]:\\[^:<>"|?\n]*|\/[^:<>"|?\n]*)/g;
	const urls = prompt.trim().match(urlRegex) || [];

	return urls.filter((url) => {
		const extensionIndex = url.lastIndexOf('.');
		if (extensionIndex === -1) {
			// If no extension (which can be the case for a directory URL), return true:
			return true;
		}
		const extension = url.slice(extensionIndex);
		return extension;
	});
}

async function indexFromDocument(
	doc: Document<Metadata>,
): Promise<(QueryEngine & RetrieverQueryEngine) | undefined> {
	try {
		const indexDB = await VectorStoreIndex.fromDocuments([doc]);
		return indexDB.asQueryEngine() as unknown as QueryEngine &
			RetrieverQueryEngine;
	} catch (error) {
		logger(error);
	}
}

const updateHistory = async (
	messageHistory: ChatMessage[],
	messageContent: MessageContentDetail[],
	document: DocumentWithIndex | undefined,
	userPrompt: string,
) => {
	messageHistory.push({
		role: 'user',
		content: messageContent,
	});

	const results = await document?.indexFromDocument?.query({
		query: removePathsInPrompt(userPrompt),
	});

	messageHistory.push({
		role: 'system',
		content: results?.response as string,
	});
};

// openai can't read
function removePathsInPrompt(prompt: string) {
	return prompt
		.split(' ')
		.map((word: string) => {
			// This regular expression checks if 'word' could be a URL or file path
			if (!/(\.|\w+:\/)/.test(word)) {
				return word;
			}
		})
		.join(' ');
}

async function main(
	model: string,
	prompts: string[],
	properties: Record<string, unknown>,
	settings: Record<string, unknown>,
): Promise<ConnectorResponse | undefined> {
	const total = prompts.length;
	const { prompt, ...restProperties } = properties;
	const systemPrompt = (prompt ||
		config.properties.find((prop) => prop.id === 'prompt')?.value) as string;
	const messageHistory: ChatMessage[] = [
		{ role: 'system', content: [{ type: 'text', text: systemPrompt }] },
	];
	const outputs: Array<Message | ErrorCompletion> = [];

	const llm = new OpenAI({
		model,
		additionalChatOptions: { ...restProperties },
		apiKey: settings?.['API_KEY'] as string,
	});

	try {
		for (let index = 0; index < total; index++) {
			try {
				const userPrompt = prompts[index];
				const docUrls = extractDocumentUrls(userPrompt);
				const messageContent: ChatMessage['content'] = [
					{ type: 'text', text: removePathsInPrompt(userPrompt) },
				];

				for await (const docUrl of docUrls) {
					try {
						const documents = await parseDocument(docUrl);
						if (Array.isArray(documents)) {
							documents.forEach(async (doc) => {
								await updateHistory(
									messageHistory,
									messageContent,
									doc,
									userPrompt,
								);
							});
						} else {
							await updateHistory(
								messageHistory,
								messageContent,
								documents,
								userPrompt,
							);
						}
					} catch (error) {
						console.log(error);
					}
				}

				logger(messageHistory);

				const response: Message = await llm.chat({
					messages: messageHistory,
				});

				outputs.push(response);

				const assistantResponse = response.message.content || 'No response.';

				messageHistory.push({
					role: 'assistant',
					content: [{ type: 'text', text: assistantResponse as string }],
				});

				logger(response, `Response to prompt ${index + 1} of ${total}`);
			} catch (error) {
				const completionWithError = mapErrorToCompletion(error, model);
				outputs.push(completionWithError);
			}
		}
		return mapToResponse(outputs, model);
	} catch (error) {
		console.error('Error in main function:', error);
		throw error;
	}
}

export { main, config };
